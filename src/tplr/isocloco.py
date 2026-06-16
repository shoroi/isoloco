"""ISOCLoCo: Isotropic Covariance merging-based gradient aggregation optimizer."""

from typing import Optional, List, Iterable, Any, Union, TypeAlias

import torch
import torch.distributed as dist

ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[dict[str, Any]]]


class ISOCLoCo(torch.optim.Optimizer):
    """
    ISOCLoCo optimizer that aggregates gradients using ISO-C (Isotropic Covariance) merging.

    This optimizer implements the ISO-C algorithm which:
    1. Averages all gradients from all workers
    2. For 2D parameters: Computes SVD of the averaged gradient
    3. Replaces singular values with their mean (isotropic covariance)
    4. Reconstructs the merged gradient with uniform singular values
    5. For 1D parameters: Uses simple average

    Supports momentum with configurable orthogonalization order:
    - orthogonalize_object="gradient": ISO-C applied to aggregated gradients, then momentum
    - orthogonalize_object="update": Momentum first, then ISO-C applied to the momentum-based update
    """

    def __init__(
        self,
        models_params: ParamsT,
        lr: float = 1.0,
        momentum: float = 0.0,
        nesterov: bool = False,
        orthogonalize_object: str = "gradient",
        exclude_keys: Optional[List[str]] = None,
        virtual_workers: int = 1,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        """
        Initialize ISOCLoCo optimizer.

        Args:
            models_params: Iterable of model parameters for each virtual worker
            lr: Learning rate (default: 1.0)
            momentum: Momentum coefficient (default: 0.0)
            nesterov: Use Nesterov momentum (default: False)
            orthogonalize_object: Controls when ISO-C is applied:
                - "gradient" (default): ISO-C applied to aggregated gradients, then momentum
                - "update": Momentum first, then ISO-C applied to the momentum-based update
            exclude_keys: Parameter name substrings to exclude from ISO-C merging
                         (e.g., ["text_projection"]). These will use averaging instead.
            virtual_workers: Number of virtual workers per process (default: 1)
            process_group: Distributed process group (default: None)
        """
        self.models_params = [list(model_param) for model_param in models_params]
        all_params = [p for model_params in self.models_params for p in model_params]
        defaults = dict(lr=lr)
        super().__init__(all_params, defaults)

        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.orthogonalize_object = orthogonalize_object
        self.virtual_workers = virtual_workers
        self.process_group = process_group
        self.is_distributed = dist.is_initialized() and process_group is not None
        self.exclude_keys = exclude_keys or []

        # Validate orthogonalize_object
        if self.orthogonalize_object not in ["gradient", "update"]:
            raise ValueError(f"orthogonalize_object must be 'gradient' or 'update', got '{self.orthogonalize_object}'")

        # Validate nesterov requires momentum
        if self.nesterov and self.momentum <= 0:
            raise ValueError("Nesterov momentum requires momentum > 0")

        # Calculate total number of workers
        if self.is_distributed:
            self.total_workers = virtual_workers * dist.get_world_size(self.process_group)
        else:
            self.total_workers = virtual_workers

        # Single set of momentum buffers (all models sync to same params)
        self.momentum_buffers = []
        if self.momentum > 0:
            for p in self.models_params[0]:  # Use first model as template
                if p.requires_grad:
                    self.momentum_buffers.append(torch.zeros_like(p))
                else:
                    self.momentum_buffers.append(None)

    def _all_gather_tensor(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Gather tensor from all workers."""
        if not self.is_distributed:
            return [tensor]

        world_size = dist.get_world_size(self.process_group)
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor, group=self.process_group)
        return gathered

    def _should_apply_isoc(self, param_name: str, grad: torch.Tensor) -> bool:
        """
        Determine if ISO-C merging should be applied to this parameter.
        
        Args:
            param_name: Name or identifier of the parameter
            grad: Gradient tensor
            
        Returns:
            True if ISO-C should be applied, False otherwise
        """
        # Only apply to 2D parameters
        if grad.dim() != 2:
            return False
        
        # Check if parameter name contains any excluded keys
        for exclude_key in self.exclude_keys:
            if exclude_key in param_name:
                return False
        
        return True

    def _apply_isoc(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply ISO-C orthogonalization to a 2D tensor.

        Computes SVD and replaces singular values with their mean:
        tensor = U @ diag(S) @ Vᵀ  →  U @ diag(mean(S)) @ Vᵀ

        Args:
            tensor: 2D tensor to orthogonalize

        Returns:
            Orthogonalized tensor
        """
        U, S, Vh = torch.linalg.svd(tensor, full_matrices=False)
        S_mean = torch.ones_like(S) * S.mean()
        return torch.linalg.multi_dot((U, torch.diag(S_mean), Vh))

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure: Optional closure that reevaluates the model and returns the loss
        """
        if closure is not None:
            closure()

        lr = self.lr

        # Process each parameter across all virtual workers
        for p_idx in range(len(self.models_params[0])):
            grads_local = []
            param_name = f"param_{p_idx}"  # Default name

            # Collect gradients from all virtual workers on this rank
            for vmodel_idx, vmodel_params in enumerate(self.models_params):
                p = vmodel_params[p_idx]
                if not p.requires_grad or p.grad is None:
                    continue

                grads_local.append(p.grad)

                # Try to get parameter name if available
                if hasattr(p, 'param_name'):
                    param_name = p.param_name

            if not grads_local:
                continue

            # Stack local virtual worker gradients
            local_stacked = torch.stack(grads_local, dim=0)  # (V, *shape)

            # Gather from all distributed workers
            gathered = self._all_gather_tensor(local_stacked)  # List of (V, *shape)

            # Flatten to list of all gradients from all workers
            all_grads = []
            for worker_grads in gathered:
                for v in range(worker_grads.shape[0]):
                    all_grads.append(worker_grads[v])

            # Compute update based on orthogonalize_object setting
            if self.orthogonalize_object == "gradient":
                # ISO-C on averaged gradients first, then momentum
                grad_mean = torch.stack(all_grads, dim=0).mean(dim=0)
                if self._should_apply_isoc(param_name, grad_mean):
                    merged = self._apply_isoc(grad_mean)
                else:
                    merged = grad_mean

                if self.momentum > 0:
                    buf = self.momentum_buffers[p_idx]
                    buf.mul_(self.momentum).add_(merged)
                    if self.nesterov:
                        update = merged.add(buf, alpha=self.momentum)
                    else:
                        update = buf.clone()
                else:
                    update = merged
            elif self.orthogonalize_object == "update":
                # Average first, momentum, then ISO-C
                grad_mean = torch.stack(all_grads, dim=0).mean(dim=0)

                if self.momentum > 0:
                    buf = self.momentum_buffers[p_idx]
                    buf.mul_(self.momentum).add_(grad_mean)
                    if self.nesterov:
                        momentum_update = grad_mean.add(buf, alpha=self.momentum)
                    else:
                        momentum_update = buf.clone()
                else:
                    momentum_update = grad_mean

                # Apply ISO-C to momentum-based update
                if self._should_apply_isoc(param_name, momentum_update):
                    update = self._apply_isoc(momentum_update)
                else:
                    update = momentum_update

            # Apply update to all models on this rank
            for vmodel_idx, vmodel_params in enumerate(self.models_params):
                p = vmodel_params[p_idx]
                if not p.requires_grad or p.grad is None:
                    continue

                p.grad = update.clone()

                # Update parameters: θ ← θ - lr * grad
                p.data.add_(p.grad, alpha=-lr)
