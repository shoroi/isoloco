# Standard library
import contextlib
from abc import ABC, abstractmethod

# Third party
import torch
import torch.distributed as dist

# Local
import tplr


class InnerOuterStrategy(ABC):
    @abstractmethod
    def inner_step(self, model, loader, inner_optimizer=None, inner_scheduler=None):
        """Execute inner optimization step"""
        pass

    @abstractmethod
    def outer_step(self, models, optimizer, scheduler=None):
        """Execute outer optimization step"""
        pass


class SimpleAccum(InnerOuterStrategy):
    def __init__(self, device, world_size, global_rank, tokenizer, config):
        self.device = device
        self.world_size = world_size
        self.global_rank = global_rank
        self.tokenizer = tokenizer
        self.config = config
        self.handles_own_communication = self.config.outer_optimizer in [
            "mergeloco",
            "tsvloco",
            "isocloco",
            "isoctsloco",
        ]

    def inner_step(self, model, loader, inner_optimizer=None, inner_scheduler=None):
        total_loss = 0
        batch_tokens = 0
        batch_count = 0
        accum_batch_size = 0

        for i, batch in enumerate(loader):
            input_ids = batch.to(self.device, non_blocking=True)
            labels = input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

            accum_batch_size += len(batch)

            with torch.amp.autocast(
                device_type=self.device.type, dtype=torch.bfloat16
            ):
                outputs = model(input_ids=input_ids, labels=labels)

            loss = outputs.loss / (
                self.config.batch_size / self.config.micro_batch_size
            )
            loss.backward()

            total_loss += loss.item()
            batch_count += 1
            batch_tokens += (labels != -100).sum().item()

            if self.global_rank == 0 and i % 20 == 0:
                tplr.logger.info(
                    f"Batch {i}, loss: {outputs.loss.item():.4f}, accum: {accum_batch_size}/{self.config.batch_size}"
                )

            if accum_batch_size >= self.config.batch_size:
                break

        return {
            "total_loss": total_loss,
            "loss_after_gather": total_loss,
            "batch_count": batch_count,
            "batch_tokens": batch_tokens,
        }


    def outer_step(self, models, optimizer, scheduler):
        model = models[0]
        actual_model = model.module if hasattr(model, "module") else model

        # Accumulate gradients from all virtual models into the first model
        if len(models) > 1:
            master_params = list(actual_model.parameters())
            for param_idx, param in enumerate(master_params):
                if param.grad is not None:
                    for vmodel in models[1:]:
                        vmodel_actual = vmodel.module if hasattr(vmodel, "module") else vmodel
                        vparam = list(vmodel_actual.parameters())[param_idx]
                        if vparam.grad is not None:
                            param.grad.add_(vparam.grad)
                    param.grad.div_(len(models))

        # All-reduce across physical GPUs (for optimizers that don't handle it internally)
        if self.world_size > 1 and not self.handles_own_communication:
            for param in actual_model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

        optimizer.step()
        if scheduler:
            scheduler.step()

        # Sync updated parameters to all virtual models
        if len(models) > 1:
            base_sd = self._unwrap(models[0]).state_dict()
            for vmodel in models[1:]:
                self._unwrap(vmodel).load_state_dict(base_sd, strict=True)

    def _unwrap(self, m):
        if hasattr(m, "_orig_mod"):
            m = m._orig_mod
        return m


class Diloco(InnerOuterStrategy):
    def __init__(self, device, world_size, global_rank, tokenizer, config):
        self.device = device
        self.world_size = world_size
        self.global_rank = global_rank
        self.tokenizer = tokenizer
        self.config = config
        self.params_offloaded = None
        self.handles_own_communication = self.config.outer_optimizer in [
            "mergeloco",
            "tsvloco",
            "isocloco",
            "isoctsloco",
        ]

    def inner_step(self, model, loader, inner_optimizer, inner_scheduler):
        """ Inner step of Diloco. Performs local optimization on a single model"""
        total_loss, batch_tokens, batch_count = 0.0, 0, 0
        loss_after_gather = 0.0

        if self.params_offloaded is None:
            self.params_offloaded = self._get_offloaded_param(model)

        inner_step_count = 0
        accum_batch_size = 0
        running_loss_this_accum = 0.0

        for i, batch in enumerate(loader):
            input_ids = batch.to(self.device, non_blocking=True)
            labels = input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            accum_batch_size += len(batch)

            with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, labels=labels)

            loss = outputs.loss / (self.config.batch_size / self.config.micro_batch_size)
            loss.backward()

            lval = loss.item()
            running_loss_this_accum += lval
            total_loss += lval
            batch_count += 1
            batch_tokens += (labels != -100).sum().item()

            if accum_batch_size >= self.config.batch_size:
                if inner_step_count == 0:
                    loss_after_gather = running_loss_this_accum
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                inner_optimizer.step()
                inner_scheduler.step()
                inner_optimizer.zero_grad(set_to_none=True)

                if self.global_rank == 0 and inner_step_count % 5 == 0:
                    tplr.logger.info(
                        f"Inner Step {inner_step_count + 1}/{self.config.inner_steps}, loss: {outputs.loss.item():.4f}"
                    )

                inner_step_count += 1
                accum_batch_size = 0
                running_loss_this_accum = 0.0

                if inner_step_count >= self.config.inner_steps:
                    break

        return {
            "total_loss": total_loss / max(1, inner_step_count),
            "loss_after_gather": loss_after_gather,
            "batch_count": batch_count,
            "batch_tokens": batch_tokens,
        }

    def outer_step(self, models, optimizer, scheduler=None):
        if self.params_offloaded is None:
            raise ValueError("No offloaded parameters found. Run inner_step first.")

        master_params = list(models[0].parameters())
        vmodels_params = [list(vmodel.parameters()) for vmodel in models[1:]]

        for param_idx, param_offloaded in enumerate(self.params_offloaded):
            # 1. Calculate gradients. If Diloco baseline, aggregate gradients to master
            saved_param = param_offloaded.to(self.device)
            master_params[param_idx].grad = saved_param - master_params[param_idx].data
            for vmodel_params in vmodels_params:
                if self.handles_own_communication:
                    vmodel_params[param_idx].grad = saved_param - vmodel_params[param_idx].data
                else:
                    master_params[param_idx].grad.add_(saved_param - vmodel_params[param_idx].data)
            
            # 2. All reduce on the gradients of vmodel[0]s across all GPUs for Diloco baseline
            if not self.handles_own_communication:
                master_params[param_idx].grad.div_(len(models))
                if self.world_size > 1:
                    dist.all_reduce(master_params[param_idx].grad, op=dist.ReduceOp.AVG)

            master_params[param_idx].data.copy_(saved_param)
            if self.handles_own_communication:
                for vmodel_params in vmodels_params:
                    vmodel_params[param_idx].data.copy_(saved_param)

        # 3. Outer optimizer step to update vmodel[0]s
        optimizer.step()

        # 4. Copy updated parameters of vmodel[0] to all other virtual models on same gpu for Diloco baseline
        if not self.handles_own_communication:
            base_sd = self._unwrap(models[0]).state_dict()
            for vmodel in models[1:]:
                self._unwrap(vmodel).load_state_dict(base_sd, strict=True)
        
        # 5. Set params_offloaded to None so it will be reloaded next inner step
        self.params_offloaded = None

    def _get_offloaded_param(self, model):
        actual_model = model.module if hasattr(model, "module") else model
        return [param.data.detach().clone().to("cpu") for param in actual_model.parameters()]

    def _unwrap(self, m):
        if hasattr(m, "_orig_mod"):
            m = m._orig_mod
        return m
