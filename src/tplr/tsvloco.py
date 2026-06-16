"""TSVLoCo: Task Singular Value merging-based gradient aggregation optimizer."""

from typing import Optional, List, Iterable, Any, Union, TypeAlias

import torch
import torch.distributed as dist

ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[dict[str, Any]]]


class TSVLoCo(torch.optim.Optimizer):
    """
    TSVLoCo optimizer that aggregates gradients using SVD-based merging.
    
    This optimizer implements the TSV-Merge algorithm which:
    1. Computes SVD of each gradient and retains a fraction of singular components
    2. Concatenates the reduced singular matrices
    3. Optionally decorrelates by computing SVD of concatenated matrices
    4. Reconstructs the merged gradient from the (optionally orthogonalized) components
    """

    def __init__(
        self,
        models_params: ParamsT,
        lr: float = 1.0,
        singular_ratio: Optional[float] = None,
        decorrelate: bool = True,
        virtual_workers: int = 1,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        """
        Initialize TSVLoCo optimizer.
        
        Args:
            models_params: Iterable of model parameters for each virtual worker
            lr: Learning rate (default: 1.0)
            singular_ratio: Fraction of singular components to retain (default: 1/num_workers)
            decorrelate: Whether to orthogonalize concatenated matrices via SVD (default: True)
            virtual_workers: Number of virtual workers per process (default: 1)
            process_group: Distributed process group (default: None)
        """
        self.models_params = [list(model_param) for model_param in models_params]
        all_params = [p for model_params in self.models_params for p in model_params]
        defaults = dict(lr=lr)
        super().__init__(all_params, defaults)

        self.lr = lr
        self.virtual_workers = virtual_workers
        self.process_group = process_group
        self.is_distributed = dist.is_initialized() and process_group is not None
        
        # Calculate total number of workers
        if self.is_distributed:
            total_workers = virtual_workers * dist.get_world_size(self.process_group)
        else:
            total_workers = virtual_workers
        
        # Default singular_ratio is 1/T where T is total number of workers
        self.singular_ratio = singular_ratio if singular_ratio is not None else 1.0 / total_workers
        self.decorrelate = decorrelate

    def _all_gather_tensor(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Gather tensor from all workers."""
        if not self.is_distributed:
            return [tensor]

        world_size = dist.get_world_size(self.process_group)
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor, group=self.process_group)
        return gathered

    def _svd_low_rank(self, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute SVD and retain only top singular_ratio components.
        
        Args:
            grad: 2D gradient tensor of shape (m, n)
        
        Returns:
            U_k: Left singular vectors, shape (m, k)
            S_k: Singular values, shape (k,)
            V_k: Right singular vectors, shape (n, k)
            
        Where k = max(1, int(singular_ratio * min(m, n)))
        """
        # Compute full SVD: grad = U @ diag(S) @ V^T
        U, S, Vh = torch.linalg.svd(grad, full_matrices=False)
        
        # Determine number of components to keep (1/T fraction)
        k = max(1, int(self.singular_ratio * min(grad.shape)))
        
        # Retain top k components
        U_k = U[:, :k]      # (m, k)
        S_k = S[:k]         # (k,)
        V_k = Vh[:k, :].T   # (n, k) - transpose Vh to get V
        
        return U_k, S_k, V_k

    def _merge_gradients_svd(self, gradients_list: List[torch.Tensor]) -> torch.Tensor:
        """
        SVD-based merging with optional decorrelation.
        
        Implements the TSV-Merge algorithm:
        1. Compute SVD for each gradient: Δᵢ = UᵢΣᵢVᵢᵀ
        2. Retain first 1/T singular components
        3. Concatenate: U = [U₁|U₂|...|Uₜ], Σ = block-diag(Σ₁,...,Σₜ), V = [V₁|V₂|...|Vₜ]
        4. If decorrelate: Compute SVD of U and V, obtain orthogonal matrices
        5. Reconstruct: M = U_⊥ Σ V_⊥ᵀ (or U Σ Vᵀ if not decorrelated)
        
        Args:
            gradients_list: List of gradient tensors from all workers
            
        Returns:
            Merged gradient tensor
        """
        if len(gradients_list) == 1:
            return gradients_list[0]
        
        # Check if this is a 2D parameter
        is_2d = gradients_list[0].dim() == 2
        
        if not is_2d:
            # For 1D parameters, use simple averaging
            return torch.stack(gradients_list, dim=0).mean(dim=0)
        
        # Step 1 & 2: Compute SVD for each gradient and retain top components
        U_list = []
        S_list = []
        V_list = []
        
        for grad in gradients_list:
            U_k, S_k, V_k = self._svd_low_rank(grad)
            U_list.append(U_k)
            S_list.append(S_k)
            V_list.append(V_k)
        
        # Step 3: Concatenate U and V matrices horizontally
        # U = [U₁ | U₂ | ... | Uₜ]
        U_concat = torch.cat(U_list, dim=1)  # (m, k*T)
        V_concat = torch.cat(V_list, dim=1)  # (n, k*T)
        
        # Create block diagonal Σ from individual Σᵢ
        total_k = sum(len(S) for S in S_list)
        Sigma = torch.zeros(
            total_k, total_k,
            device=gradients_list[0].device,
            dtype=gradients_list[0].dtype
        )
        offset = 0
        for S in S_list:
            k = len(S)
            Sigma[offset:offset+k, offset:offset+k] = torch.diag(S)
            offset += k
        
        if self.decorrelate:
            # Step 4: Compute SVD of U and V to orthogonalize
            # U = P_U D_U Q_U^T
            # V = P_V D_V Q_V^T
            P_u, D_u, Quh = torch.linalg.svd(U_concat, full_matrices=False)
            P_v, D_v, Qvh = torch.linalg.svd(V_concat, full_matrices=False)
            
            # Step 5: Obtain orthogonal matrices
            # U_⊥ = P_U Q_U^T
            # V_⊥ = P_V Q_V^T
            U_orth = P_u @ Quh
            V_orth = P_v @ Qvh
            
            # Step 6: Reconstruct merged matrix
            # M̃ = U_⊥ Σ V_⊥^T
            merged = U_orth @ Sigma @ V_orth.T
        else:
            # Without decorrelation, just use concatenated matrices
            # M̃ = U Σ V^T
            merged = U_concat @ Sigma @ V_concat.T
        
        return merged

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

            # Collect gradients from all virtual workers on this rank
            for vmodel_idx, vmodel_params in enumerate(self.models_params):
                p = vmodel_params[p_idx]
                if not p.requires_grad or p.grad is None:
                    continue

                grads_local.append(p.grad)

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

            # Merge all gradients using SVD-based merging
            merged = self._merge_gradients_svd(all_grads)

            # Apply merged gradient to all models on this rank
            for vmodel_idx, vmodel_params in enumerate(self.models_params):
                p = vmodel_params[p_idx]
                if not p.requires_grad or p.grad is None:
                    continue

                p.grad = merged.clone()

                # Update parameters: θ ← θ - lr * grad
                p.data.add_(p.grad, alpha=-lr)
