"""MergeLoCo: Model merging-based gradient aggregation optimizer."""

from typing import Optional, List, Iterable, Any, Union, TypeAlias

import torch
import torch.distributed as dist

ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[dict[str, Any]]]


class MergeLoCo(torch.optim.Optimizer):
    """
    MergeLoCo optimizer that aggregates gradients using sparsification
    and TIES-style merging.
    """

    def __init__(
        self,
        models_params: ParamsT,
        lr: float = 1.0,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
        sparsity: float = 0.0,
        prune_method: str = "magnitude",
        prune_dimension: str = "all",
        rescale_pruned: bool = False,
        ties_disjoint: bool = False,
        virtual_workers: int = 1,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        self.models_params = [list(model_param) for model_param in models_params]
        all_params = [p for model_params in self.models_params for p in model_params]
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(all_params, defaults)

        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.sparsity = sparsity
        self.prune_method = prune_method
        self.prune_dimension = prune_dimension
        self.rescale_pruned = rescale_pruned
        self.ties_disjoint = ties_disjoint
        self.virtual_workers = virtual_workers
        self.process_group = process_group
        self.is_distributed = dist.is_initialized() and process_group is not None

        # Initialize momentum buffers if needed
        if self.momentum > 0:
            self.momentum_buffers = [
                [torch.zeros_like(p) for p in model_params]
                for model_params in self.models_params
            ]

    def _all_gather_tensor(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Gather tensor from all workers."""
        if not self.is_distributed:
            return [tensor]

        world_size = dist.get_world_size(self.process_group)
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor, group=self.process_group)
        return gathered

    def _sparsify_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        """Apply sparsification to gradient."""
        if self.sparsity <= 0.0:
            return grad

        original_grad = grad.clone() if self.rescale_pruned else None

        # Determine k based on dimension
        if self.prune_dimension == "row" and grad.dim() == 2:
            k = max(1, int((1 - self.sparsity) * grad.shape[0]))
        elif self.prune_dimension == "column" and grad.dim() == 2:
            k = max(1, int((1 - self.sparsity) * grad.shape[1]))
        else:  # "all"
            k = max(1, int((1 - self.sparsity) * grad.numel()))

        # Apply pruning
        if self.prune_method == "magnitude":
            pruned = self._prune_by_magnitude(grad, k)
        else:  # "random"
            pruned = self._prune_random(grad, k)

        # Rescale if needed
        if self.rescale_pruned and original_grad is not None:
            pruned = pruned / (1 - self.sparsity)

        return pruned

    def _prune_by_magnitude(self, grad: torch.Tensor, k: int) -> torch.Tensor:
        """Magnitude-based pruning."""
        if self.prune_dimension == "row" and grad.dim() == 2:
            row_norms = torch.norm(grad, dim=1)
            _, top_idx = torch.topk(row_norms, k)
            mask = torch.zeros(grad.shape[0], dtype=torch.bool, device=grad.device)
            mask[top_idx] = True
            return grad * mask.unsqueeze(1)
        elif self.prune_dimension == "column" and grad.dim() == 2:
            col_norms = torch.norm(grad, dim=0)
            _, top_idx = torch.topk(col_norms, k)
            mask = torch.zeros(grad.shape[1], dtype=torch.bool, device=grad.device)
            mask[top_idx] = True
            return grad * mask.unsqueeze(0)
        else:  # "all"
            flat = grad.flatten()
            _, top_idx = torch.topk(torch.abs(flat), k)
            mask = torch.zeros_like(flat, dtype=torch.bool)
            mask[top_idx] = True
            return (mask * flat).reshape(grad.shape)

    def _prune_random(self, grad: torch.Tensor, k: int) -> torch.Tensor:
        """Random pruning."""
        if self.prune_dimension == "row" and grad.dim() == 2:
            idx = torch.randperm(grad.shape[0], device=grad.device)[:k]
            mask = torch.zeros(grad.shape[0], dtype=torch.bool, device=grad.device)
            mask[idx] = True
            return grad * mask.unsqueeze(1)
        elif self.prune_dimension == "column" and grad.dim() == 2:
            idx = torch.randperm(grad.shape[1], device=grad.device)[:k]
            mask = torch.zeros(grad.shape[1], dtype=torch.bool, device=grad.device)
            mask[idx] = True
            return grad * mask.unsqueeze(0)
        else:  # "all"
            flat = grad.flatten()
            idx = torch.randperm(len(flat), device=grad.device)[:k]
            mask = torch.zeros_like(flat, dtype=torch.bool)
            mask[idx] = True
            return (mask * flat).reshape(grad.shape)

    def _merge_gradients(self, gradients_list: List[torch.Tensor]) -> torch.Tensor:
        """
        TIES-style merging.

        Elect: Compute elected sign as sign(sum of all values). This weights
        by magnitude - the sign with greater total mass wins.

        Disjoint Merge: Average only values whose sign matches the elected sign,
        always ignoring zero values.

        When ties_disjoint=False, performs simple averaging without sign filtering.
        """
        stacked = torch.stack(gradients_list, dim=0)  # (N, *shape)

        if self.ties_disjoint:
            # TIES disjoint merge
            # Step 1: Elect sign based on sum (accounts for magnitude)
            elected_sign = torch.sign(stacked.sum(dim=0))

            # Step 2: Find values matching elected sign
            # Note: sign(0) = 0, so zeros naturally don't match +1 or -1,
            # which means zeros are always ignored as required by TIES
            signs = torch.sign(stacked)
            sign_match = signs == elected_sign.unsqueeze(0)

            # Step 3: Average only matching values
            masked = stacked * sign_match
            matching_count = sign_match.sum(dim=0).clamp(min=1).float()
            merged = masked.sum(dim=0) / matching_count
        else:
            # Simple averaging without sign filtering
            merged = stacked.mean(dim=0)

        return merged

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            closure()

        lr = self.lr

        for p_idx in range(len(self.models_params[0])):
            grads_local = []

            for vmodel_idx, vmodel_params in enumerate(self.models_params):
                p = vmodel_params[p_idx]
                if not p.requires_grad or p.grad is None:
                    continue

                # Apply weight decay (decoupled/AdamW style)
                if self.weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * self.weight_decay)

                # Sparsify gradient
                sparse_grad = self._sparsify_gradient(p.grad)
                grads_local.append(sparse_grad)

            if not grads_local:
                continue

            # Stack local virtual worker gradients and all-gather
            local_stacked = torch.stack(grads_local, dim=0)  # (V, *shape)
            gathered = self._all_gather_tensor(local_stacked)  # list of (V, *shape)

            # Flatten to list of all gradients from all workers
            all_grads = []
            for worker_grads in gathered:
                for v in range(worker_grads.shape[0]):
                    all_grads.append(worker_grads[v])

            # Merge all gradients
            merged = self._merge_gradients(all_grads)

            # Apply to all models on this rank
            for vmodel_idx, vmodel_params in enumerate(self.models_params):
                p = vmodel_params[p_idx]
                if not p.requires_grad or p.grad is None:
                    continue

                p.grad = merged.clone()

                # Apply momentum if enabled
                if self.momentum > 0:
                    buf = self.momentum_buffers[vmodel_idx][p_idx]
                    buf.mul_(self.momentum).add_(p.grad)
                    if self.nesterov:
                        p.grad = p.grad.add(buf, alpha=self.momentum)
                    else:
                        p.grad = buf.clone()

                # Update parameters
                p.data.add_(p.grad, alpha=-lr)
