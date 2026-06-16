"""ISOCTSLoCo: Isotropic Covariance with Common + Task-Specific space merging optimizer."""

from typing import Optional, List, Iterable, Any, Union, TypeAlias

import torch
import torch.distributed as dist

ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[dict[str, Any]]]


class ISOCTSLoCo(torch.optim.Optimizer):
    """
    ISOCTSLoCo optimizer that aggregates gradients using ISO-CTS merging.

    Extends ISO-C by decomposing the gradient space into:
    - Common space: dimensions shared across all workers (from SVD of summed gradients)
    - Task-specific spaces: per-worker dimensions (from SVD of residuals after removing common space)

    For 2D parameters:
    1. Sum all worker gradients and compute SVD to find the common space
    2. For each worker: subtract common space projection, SVD the residual for task-specific components
    3. Combine common + task-specific U, S, V matrices
    4. Orthogonalize combined U and V via SVD
    5. Replace singular values with their mean (isotropic)
    6. Reconstruct the merged gradient

    For 1D parameters: simple average.

    Supports optional momentum (applied after ISO-CTS merging).
    """

    def __init__(
        self,
        models_params: ParamsT,
        lr: float = 1.0,
        momentum: float = 0.0,
        nesterov: bool = False,
        common_space_fraction: float = 0.8,
        exclude_keys: Optional[List[str]] = None,
        virtual_workers: int = 1,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        """
        Initialize ISOCTSLoCo optimizer.

        Args:
            models_params: Iterable of model parameters for each virtual worker
            lr: Learning rate (default: 1.0)
            momentum: Momentum coefficient (default: 0.0)
            nesterov: Use Nesterov momentum (default: False)
            common_space_fraction: Fraction of SVD dimensions allocated to common space (default: 0.8)
            exclude_keys: Parameter name substrings to exclude from ISO-CTS merging
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
        self.common_space_fraction = common_space_fraction
        self.virtual_workers = virtual_workers
        self.process_group = process_group
        self.is_distributed = dist.is_initialized() and process_group is not None
        self.exclude_keys = exclude_keys or []

        if self.nesterov and self.momentum <= 0:
            raise ValueError("Nesterov momentum requires momentum > 0")

        if self.is_distributed:
            self.total_workers = virtual_workers * dist.get_world_size(self.process_group)
        else:
            self.total_workers = virtual_workers

        # Single set of momentum buffers (all models sync to same params)
        self.momentum_buffers = []
        if self.momentum > 0:
            for p in self.models_params[0]:
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

    def _should_apply_isocts(self, param_name: str, grad: torch.Tensor) -> bool:
        """Determine if ISO-CTS merging should be applied to this parameter."""
        if grad.dim() != 2:
            return False
        for exclude_key in self.exclude_keys:
            if exclude_key in param_name:
                return False
        return True

    def _apply_isocts(self, all_grads: List[torch.Tensor]) -> torch.Tensor:
        """
        Apply ISO-CTS merging to a list of 2D gradient tensors.

        Decomposes into common + task-specific spaces, orthogonalizes,
        and reconstructs with isotropic singular values.

        Args:
            all_grads: List of 2D gradient tensors, one per worker

        Returns:
            Merged gradient tensor
        """
        num_tasks = len(all_grads)
        all_grads = [g / num_tasks for g in all_grads]
        shape_ = all_grads[0].shape
        min_dim = min(shape_)

        # 1. Sum all gradients to find common space
        combined_w = sum(all_grads)

        # 2. Calculate common space size, ensuring task-specific space divides evenly
        common_space_index_s = int(min_dim * self.common_space_fraction)
        task_specific_total = round((min_dim - common_space_index_s) / num_tasks) * num_tasks
        common_space_index_s = min_dim - task_specific_total

        # 3. SVD of summed gradients -> extract common space
        u, s, v = torch.linalg.svd(combined_w, full_matrices=False)
        common_space_u = u[:, :common_space_index_s]
        common_space_s = s[:common_space_index_s]
        common_space_v = v[:common_space_index_s, :]

        # 4. Extract task-specific spaces
        n_dims_per_task = int((min_dim - common_space_index_s) / num_tasks) if num_tasks > 0 else 0

        # Initialize combined space matrices using full SVD output as template
        combined_space_u = torch.zeros_like(u)
        combined_space_s = torch.zeros_like(s)
        combined_space_v = torch.zeros_like(v)

        for i, grad in enumerate(all_grads):
            # Remove common space projection
            w_ts = grad - common_space_u @ common_space_u.T @ grad
            u_ts, s_ts, v_ts = torch.linalg.svd(w_ts, full_matrices=False)

            # Store top task-specific components
            combined_space_u[:, i * n_dims_per_task : (i + 1) * n_dims_per_task] = u_ts[:, :n_dims_per_task]
            combined_space_s[i * n_dims_per_task : (i + 1) * n_dims_per_task] = s_ts[:n_dims_per_task]
            combined_space_v[i * n_dims_per_task : (i + 1) * n_dims_per_task, :] = v_ts[:n_dims_per_task, :]

        # 5. Insert common space components after task-specific ones
        cs_start = num_tasks * n_dims_per_task
        combined_space_u[:, cs_start : cs_start + common_space_index_s] = common_space_u
        combined_space_s[cs_start : cs_start + common_space_index_s] = common_space_s
        combined_space_v[cs_start : cs_start + common_space_index_s, :] = common_space_v

        # 6. Orthogonalize combined U and V
        u_cu, _s_cu, v_cu = torch.linalg.svd(combined_space_u, full_matrices=False)
        combined_space_u = u_cu @ v_cu

        u_cv, _s_cv, v_cv = torch.linalg.svd(combined_space_v, full_matrices=False)
        combined_space_v = u_cv @ v_cv

        # 7. Isotropic singular values
        combined_space_s = torch.ones_like(combined_space_s) * combined_space_s.mean()

        # 8. Reconstruct
        return torch.linalg.multi_dot((combined_space_u, torch.diag(combined_space_s), combined_space_v))

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        if closure is not None:
            closure()

        lr = self.lr

        for p_idx in range(len(self.models_params[0])):
            grads_local = []
            param_name = f"param_{p_idx}"

            for vmodel_idx, vmodel_params in enumerate(self.models_params):
                p = vmodel_params[p_idx]
                if not p.requires_grad or p.grad is None:
                    continue
                grads_local.append(p.grad)
                if hasattr(p, 'param_name'):
                    param_name = p.param_name

            if not grads_local:
                continue

            # Stack local virtual worker gradients
            local_stacked = torch.stack(grads_local, dim=0)  # (V, *shape)

            # Gather from all distributed workers
            gathered = self._all_gather_tensor(local_stacked)  # List of (V, *shape)

            # Flatten to list of all individual worker gradients
            all_grads = []
            for worker_grads in gathered:
                for v in range(worker_grads.shape[0]):
                    all_grads.append(worker_grads[v])

            # ISO-CTS on per-worker gradients, then momentum
            if self._should_apply_isocts(param_name, all_grads[0]):
                merged = self._apply_isocts(all_grads)
            else:
                merged = torch.stack(all_grads, dim=0).mean(dim=0)

            if self.momentum > 0:
                buf = self.momentum_buffers[p_idx]
                buf.mul_(self.momentum).add_(merged)
                if self.nesterov:
                    update = merged.add(buf, alpha=self.momentum)
                else:
                    update = buf.clone()
            else:
                update = merged

            # Apply update to all models on this rank
            for vmodel_idx, vmodel_params in enumerate(self.models_params):
                p = vmodel_params[p_idx]
                if not p.requires_grad or p.grad is None:
                    continue
                p.grad = update.clone()
                p.data.add_(p.grad, alpha=-lr)
