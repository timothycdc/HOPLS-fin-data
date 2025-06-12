#------------ NOTE: -------------------#
# This code is an unreleased work of HOPLS-MILR-RHOOI.
# It is not part of the FYP. It is too comple and has numerical issues

import torch
import tensorly as tl
from tensorly import tucker_to_tensor, fold
from tensorly.decomposition import tucker
from tensorly.tenalg import kronecker, mode_dot, multi_mode_dot
from tensorly.base import unfold 
from typing import List, Optional, Tuple, Union

# Use PyTorch backend
tl.set_backend("pytorch")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


def matricize_n(tensor: torch.Tensor, mode: int = 0) -> torch.Tensor:
    """Unfolds a tensor into a matrix along a specified mode."""
    return unfold(tensor, mode)

def qsquared_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Computes Q² score.
    PRESS = sum of squared errors
    TSS = total sum of squares (variance of y_true * N)
    Q² = 1 - PRESS/TSS
    """
    y_true_flat = y_true.contiguous().view(y_true.shape[0], -1)
    y_pred_flat = y_pred.contiguous().view(y_pred.shape[0], -1)
    
    press = torch.norm(y_true_flat - y_pred_flat) ** 2
    # TSS centered around the mean of y_true
    tss = torch.norm(y_true_flat - torch.mean(y_true_flat, dim=0, keepdim=True)) ** 2

    if tss.abs() < 1e-12: # Avoid division by zero if y_true is constant or zero
        # If y_true is constant, and y_pred is also that constant, press is 0, Q2 is 1.
        # If y_true is constant, and y_pred is different, press > 0, Q2 < 0.
        return 1.0 if press.abs() < 1e-12 else -float('inf')
    return float(1 - press / tss)


def _construct_milr_weight_tensor(
    core_shape: Tuple[int, ...],
    mode_ranks_for_core_modes: List[int], 
    alpha: float,
    num_loading_matrices_for_core: int, # This argument seems redundant based on current logic using len(active_core_shape)
    dev: torch.device
) -> torch.Tensor:
    """
    Constructs the Mean Index-Weighted L2 regularization weight tensor W_G or W_D.
    core_shape: Shape of the core tensor (e.g., (1, L2, L3, ..., LN)).
    mode_ranks_for_core_modes: List of ranks [L2, L3, ..., LN] or [K2, ..., KM].
                               These are the L_{j+1} values from the formula.
                               Indices correspond to the core modes *after* the first sample mode.
                               E.g., for G (1, L2, L3), ranks are [L2, L3] for core modes 1, 2.
    alpha: Weighting exponent.
    num_loading_matrices_for_core: N-1 for G, M-1 for D. (Argument kept for potential consistency, but logic uses active_core_shape len)
    """
    # Handle scalar core cases where there are no feature modes to penalize based on index.
    # For a scalar core (shape (1,)), the MILR penalty should reduce to standard ridge (weight 1).
    # The formula involves a sum over feature modes of the core. If there are none, the sum is empty (0).
    # Division by num_loading_matrices_for_core (which is 0 for N-1=0 or M-1=0) is division by zero.
    # Standard ridge on a scalar x^2 is lambda * x^2. The penalty element-wise formula is x_ls / (1 + lambda*w).
    # For standard ridge, w should be 1.0.
    # Let's handle scalar core (shape (1,)) case explicitly to apply weight 1.0.
    if core_shape == (1,):
         return torch.ones(core_shape, device=dev, dtype=torch.float64)
         
    # Handle empty core shape (e.g., from zero ranks) - should return zeros
    if not core_shape or 0 in core_shape:
         return torch.zeros(core_shape, device=dev, dtype=torch.float64)


    active_core_shape = core_shape[1:] # Modes corresponding to L2...LN or K2...KM
    num_factors_for_this_core = len(active_core_shape) # Correctly determine the number of feature modes in the core

    # Basic sanity check: number of ranks provided should match the number of active core dimensions
    if len(mode_ranks_for_core_modes) != num_factors_for_this_core:
        raise ValueError(f"Number of active core dimensions ({num_factors_for_this_core}) does not match number of ranks provided ({len(mode_ranks_for_core_modes)}) for MILR weights.")

    try:
        # meshgrid over the active core shape dimensions
        mesh_idx_grids = torch.meshgrid(*[torch.arange(s, device=dev) for s in active_core_shape], indexing='ij')
    except RuntimeError as e: 
        print(f"Error in meshgrid for active_core_shape={active_core_shape}")
        raise e

    sum_normalized_indices_powered = torch.zeros(active_core_shape, device=dev, dtype=torch.float64)

    # Loop over the active core dimensions (corresponding to l_2, l_3, ..., l_N etc.)
    # The i-th dimension of active_core_shape corresponds to the i-th loading matrix
    # P^(i+1) or Q^(i+1) and its index l_{i+2} in the original notation.
    # The rank for this mode is mode_ranks_for_core_modes[i].
    for i in range(num_factors_for_this_core):
        one_based_indices = mesh_idx_grids[i].double() + 1.0 # Convert 0-based torch index to 1-based for formula
        rank_for_current_mode = float(mode_ranks_for_core_modes[i])
        
        if rank_for_current_mode <= 0: # Handle zero or negative ranks defensively
            term = torch.zeros_like(one_based_indices)
        else:
            term = (one_based_indices / rank_for_current_mode) ** alpha
        sum_normalized_indices_powered += term
        
    # The formula averages the normalized powered indices across the relevant modes
    # The number of modes summed over is num_factors_for_this_core
    if num_factors_for_this_core > 0:
        weights_for_active_modes = (1.0 / num_factors_for_this_core) * sum_normalized_indices_powered
        final_weights = weights_for_active_modes.unsqueeze(0) # Add back the first mode of size 1
    else: # This case should be caught by the initial scalar check, but defensive
         final_weights = torch.ones(core_shape, device=dev, dtype=torch.float64) # Default to weight 1 for scalar

    return final_weights


def _rhooi_orth(
    tensor: torch.Tensor,
    ranks: List[int],
    penalties: List[float], # Penalty lambda for each mode of the *tensor* being decomposed
    init: str = "svd", # Keep init parameter for potential future expansion (e.g. random init)
    n_iter_max: int = 100,
    tol: float = 1e-7,
    verbose: bool = False,
    dev: torch.device = device,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Ridge-Regularised HOOI with Orthogonal Loadings (RHOOI-Orth) using QR decomposition.
    Decomposes tensor T into G x_1 P_1 ... x_K P_K, minimizing ||T - [|G; P_1..P_K|]||^2 + sum lambda_k ||P_k||^2
    while enforcing P_k^T P_k = I.

    Args:
        tensor: The input tensor (e.g., Cr).
        ranks: List of target ranks [L1, L2, ..., LK] for each mode of the tensor.
        penalties: List of lambda penalties [lambda_1, lambda_2, ..., lambda_K] for each mode's loading matrix.
                   Must have the same length as ranks and tensor.ndim.
        init: Initialization method ("svd" is implemented).
        n_iter_max: Maximum number of ALS iterations.
        tol: Convergence tolerance for the objective function.
        verbose: Print convergence info.
        dev: Device to use.

    Returns:
        core: The core tensor G.
        factors: List of orthonormal loading matrices [P1, P2, ..., PK].
    """
    if len(ranks) != tensor.ndim:
        raise ValueError(f"Number of ranks ({len(ranks)}) must match tensor number of modes ({tensor.ndim})")
    if len(penalties) != tensor.ndim:
         raise ValueError(f"Number of penalties ({len(penalties)}) must match tensor number of modes ({tensor.ndim})")
    if init != "svd":
        # Only SVD initialization is implemented currently
        raise ValueError("Only 'svd' initialization is supported for RHOOI-Orth.")


    # --- Manual SVD-based Initialization ---
    factors: List[torch.Tensor] = []
    for k in range(tensor.ndim):
        if ranks[k] == 0:
             factors.append(torch.empty((tensor.shape[k], 0), device=dev, dtype=tensor.dtype))
             continue

        # Get mode-k matricization
        tensor_unfolded = unfold(tensor, mode=k) # Shape: (I_k, product(I_j for j!=k))

        # Compute SVD and take first ranks[k] left singular vectors
        # Use full_matrices=False for economic SVD
        U, _, _ = torch.linalg.svd(tensor_unfolded, full_matrices=False)

        # Take the first ranks[k] columns (corresponding to largest singular values)
        factors.append(U[:, :ranks[k]])
    # --- End Initialization ---


    obj_hist = []

    for iter_num in range(n_iter_max):
        # --- Start of Outer Loop Iteration ---

        # Calculate the core tensor ONCE at the beginning of the outer iteration
        # using the factors from the *previous* iteration or initialization.
        # This core will be used for updating all factors in this iteration.
        current_core = multi_mode_dot(tensor, [f.T for f in factors], modes=list(range(tensor.ndim)))
        core_unfolds = [unfold(current_core, mode=i) for i in range(tensor.ndim)]
        
        

        for k in range(tensor.ndim):
            # --- Start of Inner Loop (Update factors[k]) ---
            
            # Project tensor onto subspaces of *other* factors (using factors from *previous* iteration/step)
            modes_except_k = [i for i in range(tensor.ndim) if i != k]
            
            if not modes_except_k: # Tensor is 1st order (vector) - should not happen in typical Cr case, but handle defensively
                 T_proj_k = tensor.clone() # No projection needed
            else:
                 # Need factors in correct order for multi_mode_dot modes argument
                 factors_for_projection = [factors[i].T for i in modes_except_k]
                 T_proj_k = multi_mode_dot(tensor, factors_for_projection, modes=modes_except_k)

            # Matricize the projected tensor along mode k
            T_proj_k_matricized = unfold(T_proj_k, mode=k) # Shape (I_k, product(L_j for j!=k))

            # Matricize the *current_core* (calculated ONCE per outer loop) along mode k
            # This is the G_k_mat used in the ridge formula
            # G_k_mat = unfold(current_core, mode=k) # Shape (L_k, product(L_j for j!=k))
            G_k_mat = core_unfolds[k]       

            # Calculate the update matrix A = T_proj_k_matricized @ G_k_mat.T @ (G_k_mat @ G_k_mat.T + lambda_k I)^-1
            lambda_k = penalties[k]
            
            if ranks[k] == 0: # Handle rank 0 case explicitly
                 A = torch.empty((tensor.shape[k], 0), device=dev, dtype=tensor.dtype)
            else:
                # identity_Lk = torch.eye(ranks[k], device=dev, dtype=tensor.dtype)
                # inv_term = G_k_mat @ G_k_mat.T + lambda_k * identity_Lk

                # # Calculate B = T_proj_k_matricized @ G_k_mat.T
                # B_solve_T = G_k_mat @ T_proj_k_matricized.T # Shape (Lk, Ik)

                # # Solve inv_term^T @ X_transpose = B_solve_T for X_transpose. X_transpose will have shape (Lk, Ik).
                # X_transpose = torch.linalg.solve(inv_term.T, B_solve_T)
                # A = X_transpose.T # Shape (Ik, Lk)
                
                # inv_term = G_k_mat @ G_k_mat.T + λ I is symmetric positive-definite.
                # Replace the generic solve with a Cholesky factorisation:
                identity = torch.eye(ranks[k], device=dev, dtype=tensor.dtype)
                gram     = G_k_mat @ G_k_mat.T + lambda_k * identity          # SPD
                chol     = torch.linalg.cholesky(gram)                         # L  (Ik×Ik)
                # solves L Lᵀ X = B  faster than solve()
                A        = torch.cholesky_solve((G_k_mat @ T_proj_k_matricized.T), chol).T


            # Orthogonalize A using QR decomposition
            # A has shape (I_k, L_k)
            if A.shape[1] == 0: # Handle case where rank was 0
                 factors[k] = torch.empty((A.shape[0], 0), device=dev, dtype=A.dtype)
            else:
                 Q, R = torch.linalg.qr(A, mode='reduced') # 'reduced' ensures Q is size I_k x L_k
                 factors[k] = Q # New P_k is Q
            # --- End of Inner Loop ---

        # Recalculate objective for convergence using the core and factors from this outer iteration
        # The core 'current_core' was calculated at the start of the outer loop.
        # The factors 'factors' were updated within the inner loop.
        reconstruction = tucker_to_tensor((current_core, factors))
        rec_error = torch.norm(tensor - reconstruction)**2
        
        # The ||P_k||_F^2 for an orthonormal matrix of size Ik x Lk is Lk.
        # Recompute the norm squared for safety, although it should be Lk.
        penalty_term = sum(penalties[k] * torch.norm(factors[k])**2 for k in range(tensor.ndim))
        
        current_obj = rec_error + penalty_term
        obj_hist.append(current_obj.item())

        if verbose:
            relative_rec_error = torch.norm(tensor - reconstruction) / torch.norm(tensor) if torch.norm(tensor) > 1e-12 else float('inf')
            print(f"RHOOI-Orth Iter {iter_num+1}: Objective = {current_obj.item():.6f}, Rec Error = {rec_error.item():.6f} (Relative: {relative_rec_error:.6f}), Penalty = {penalty_term.item():.6f}")

        if iter_num > 0 and abs(obj_hist[-1] - obj_hist[-2]) < tol * abs(obj_hist[-2]): # Relative tolerance check
            if verbose:
                print(f"RHOOI-Orth converged at iteration {iter_num+1}")
            break

    # Final core calculation using the final factors after convergence
    # This is needed for the return value, as the core 'current_core' inside the loop
    # was technically based on factors from the *start* of the last outer iteration.
    # However, given the update rules, the core corresponding to the *final* factors
    # is simply the contraction of the original tensor with the transpose of the final factors.
    final_core = multi_mode_dot(tensor, [f.T for f in factors], modes=list(range(tensor.ndim)))


    return final_core, factors

# The rest of the HOPLS_MILR_RHOOI class remains exactly the same.

class HOPLS_MILR_RHOOI:
    def __init__(
        self,
        R: int,
        Ln: Union[List[int], Tuple[int]], 
        Km: Optional[Union[List[int], Tuple[int]]] = None, 
        lambda_X: float = 1e-3, 
        lambda_Y: float = 1e-3, 
        alpha: float = 1.0, 
        lambda_P_factor_penalty: float = 1e-5, # Single penalty for all X-factor matrices (modes 2..N)
        lambda_Q_factor_penalty: float = 1e-5, # Single penalty for all Y-factor matrices (modes 2..M or mode 1 for Y-matrix)
        metric: Optional[callable] = None,
        epsilon: float = 1e-9, # For deflation convergence check
        rhooi_n_iter_max: int = 100, 
        rhooi_tol: float = 1e-7, # For RHOOI inner loop convergence
        rhooi_verbose: bool = False,
    ) -> None:
        self.R = R
        self.Ln = list(Ln)
        self.Km = list(Km) if Km is not None else None
        self.metric = metric or qsquared_score
        self.epsilon = float(epsilon)
        self.lambda_X = float(lambda_X)
        self.lambda_Y = float(lambda_Y)
        self.alpha = float(alpha)
        self.lambda_P_factor_penalty = float(lambda_P_factor_penalty)
        self.lambda_Q_factor_penalty = float(lambda_Q_factor_penalty)
        self.rhooi_n_iter_max = rhooi_n_iter_max
        self.rhooi_tol = float(rhooi_tol)
        self.rhooi_verbose = rhooi_verbose

        self._is_matrix_Y: bool = False
        self.N_modal_X: int = 0 
        self.M_modal_Y: int = 0 
        
        self.actual_Ln_used: List[int] = []
        self.actual_Km_used: List[int] = [] 

        # Model components
        self.P_r_all_components: List[List[torch.Tensor]] = [] 
        self.Q_r_all_components: Union[List[List[torch.Tensor]], torch.Tensor] = [] # List of lists for Tensor Y, Tensor for Matrix Y
        self.G_r_all_components: List[torch.Tensor] = [] 
        self.D_r_all_components: Union[List[torch.Tensor], torch.Tensor] = [] # List of tensors for Tensor Y, Tensor for Matrix Y (scalar dr)
        self.T_mat: Optional[torch.Tensor] = None 
        self.W_mat: Optional[torch.Tensor] = None # X-side weights for prediction
        self.num_components_fitted = 0

    def _fit_tensor_X_matrix_Y(self, X: torch.Tensor, Y: torch.Tensor) -> "HOPLS_MILR_RHOOI":
        self._is_matrix_Y = True
        I1_samples = X.shape[0]
        M_responses_Y = Y.shape[1]
        
        self.N_modal_X = X.ndim - 1
        self.actual_Ln_used = list(self.Ln) 
        if self.N_modal_X > 0 and len(self.actual_Ln_used) != self.N_modal_X:
             raise ValueError(f"Length of Ln ({len(self.actual_Ln_used)}) must match X.ndim-1 ({self.N_modal_X}) for tensor X")
        if self.N_modal_X == 0: # X is a vector (N_samples,)
             self.actual_Ln_used = [] # No P factors

        Er, Fr_mat = X.clone(), Y.clone()

        P_r_list_of_lists_accum: List[List[torch.Tensor]] = []
        q_r_list_accum: List[torch.Tensor] = []
        G_r_list_accum: List[torch.Tensor] = []
        d_r_list_accum: List[torch.Tensor] = [] # List of scalars (wrapped in tensor)
        t_r_list_accum: List[torch.Tensor] = [] # List of (N_samples, 1) tensors
        W_r_list_accum: List[torch.Tensor] = [] # List of (prod(X_modes_except_samples), 1) tensors
        
        num_fitted_for_loop = 0
        for _r_component_idx in range(self.R): 
            if torch.norm(Er) < self.epsilon or torch.norm(Fr_mat) < self.epsilon:
                if self.rhooi_verbose:
                    print(f"Stopping at component {_r_component_idx + 1} due to small residual norm.")
                break
            num_fitted_for_loop += 1

            Cr = torch.tensordot(Er, Fr_mat, dims=([0], [0])) # Shape (I2,...,IN, M)
            
            # Cr modes: 0..N-2 correspond to X modes 2..N -> P(1)..P(N-1)
            # Cr mode N-1 corresponds to Y mode 1 -> q
            
            ranks_for_Cr_RHOOI = self.actual_Ln_used + [M_responses_Y] # Ranks for P(1)..P(N-1) + rank for q
            penalties_for_Cr_RHOOI = [self.lambda_P_factor_penalty] * self.N_modal_X + [self.lambda_Q_factor_penalty] 
            
            # Need to handle scalar Cr case (N=1, M=1)
            if Cr.ndim == 0: # Cr is scalar (X is vector, Y is vector)
                 _Gr_C_dummy = Cr.clone().reshape(1,1) # Make it 2D for consistency, though RHOOI handles 0D
                 factors_Cr = [] # No factors
            else:
                 _Gr_C_dummy, factors_Cr = _rhooi_orth(
                     Cr, ranks_for_Cr_RHOOI, penalties_for_Cr_RHOOI,
                     init="svd", n_iter_max=self.rhooi_n_iter_max, tol=self.rhooi_tol,
                     verbose=self.rhooi_verbose, dev=device
                 )
            
            if Cr.ndim == 0: # X vector, Y vector case (N=1, M=1)
                 q_r = torch.ones((1,1), device=device, dtype=X.dtype) # q is effectively 1 (scalar Y loading)
                 P_r_current_comp = [] # No P factors
                 _Gr_C_dummy = Cr.clone().reshape(1,1) # Ensure core is shape (1,1) from scalar Cr
            elif self.N_modal_X == 0: # X vector (N=1), Y matrix (M>1) case
                 # Cr is (M,), ranks are [M], penalties are [lambda_Q]. RHOOI should return core (M,) and factor (M, M).
                 # This is not consistent with the HOPLS2 structure where q is (M, 1).
                 # The HOPLS paper's Proposition 3.4 and algorithm uses q as (M,1) corresponding to a rank-1 HOOI output.
                 # Let's adjust ranks for Cr_RHOOI: [1] + self.actual_Ln_used. Cr modes: 0 is Y, 1..N-1 are X.
                 # Original Cr shape for Tensor-Matrix: I2 x ... x IN x M. Modes 0..N-2 are X, mode N-1 is Y.
                 # The tensordot(Er, Fr_mat, dims=([0],[0])) results in (I2,...,IN, M) shape.
                 # Mode 0 of Cr is I2, ..., mode N-2 is IN, mode N-1 is M.
                 # Ranks for Cr should be L2, ..., LN, 1 for modes 0..N-2 and N-1 respectively.
                 # Penalties should match this. lambda_P for modes 0..N-2, lambda_Q for mode N-1.
                 
                 # Corrected Ranks and Penalties for Cr (I2..IN, M)
                 ranks_for_Cr_RHOOI = self.actual_Ln_used + [1] # L2...LN for modes 0..N-2, 1 for mode N-1
                 penalties_for_Cr_RHOOI = [self.lambda_P_factor_penalty] * self.N_modal_X + [self.lambda_Q_factor_penalty]

                 _Gr_C_dummy, factors_Cr = _rhooi_orth(
                     Cr, ranks_for_Cr_RHOOI, penalties_for_Cr_RHOOI,
                     init="svd", n_iter_max=self.rhooi_n_iter_max, tol=self.rhooi_tol,
                     verbose=self.rhooi_verbose, dev=device
                 )
                 # Factors from RHOOI on Cr (I2..IN, M): factors_Cr[0]..factors_Cr[N-2] are P(1)..P(N-1), factors_Cr[N-1] is q
                 P_r_current_comp = factors_Cr[:self.N_modal_X]
                 q_r = factors_Cr[self.N_modal_X] # This should be M x 1 if rank is 1
                 
            else: # N > 1, M > 1 (Tensor X, Matrix Y) - This is the standard HOPLS2 case
                 # Cr is (I2..IN, M)
                 ranks_for_Cr_RHOOI = self.actual_Ln_used + [1] # L2...LN for modes 0..N-2, 1 for mode N-1
                 penalties_for_Cr_RHOOI = [self.lambda_P_factor_penalty] * self.N_modal_X + [self.lambda_Q_factor_penalty]

                 _Gr_C_dummy, factors_Cr = _rhooi_orth(
                     Cr, ranks_for_Cr_RHOOI, penalties_for_Cr_RHOOI,
                     init="svd", n_iter_max=self.rhooi_n_iter_max, tol=self.rhooi_tol,
                     verbose=self.rhooi_verbose, dev=device
                 )
                 # Factors from RHOOI on Cr (I2..IN, M): factors_Cr[0]..factors_Cr[N-2] are P(1)..P(N-1), factors_Cr[N-1] is q
                 P_r_current_comp = factors_Cr[:self.N_modal_X]
                 q_r = factors_Cr[self.N_modal_X] # This should be M x 1 if rank is 1

            # Ensure q_r is (M, 1) and has unit norm (RHOOI-Orth applies QR which should make it unit norm if rank is 1)
            if q_r.ndim == 1: # Should be (M,) if rank is 1 in RHOOI
                 q_r = q_r.unsqueeze(1) # Make it (M, 1)
            q_r_norm = torch.norm(q_r).clamp(min=self.epsilon)
            q_r = q_r / q_r_norm # Re-normalize just to be safe, though QR should handle rank-1

            # Ensure P_r_current_comp factors are (Ii, Li) and orthonormal (RHOOI-Orth applies QR)
            # No explicit re-norm needed if RHOOI-Orth is correct

            _X_proj_for_tr = Er
            if P_r_current_comp: # Apply P factors if they exist (N > 1)
                 _X_proj_for_tr = multi_mode_dot(Er, [p.T for p in P_r_current_comp], modes=list(range(1, self.N_modal_X + 1)))
            
            # Derivation of t_r using pseudo-inverse involving the RHOOI core _Gr_C_dummy
            # This core is (L2..LN, 1) shape
            X_proj_tr_mat = unfold(_X_proj_for_tr, mode=0) # Shape (I1, L2*...*LN) or (I1, 1) if N=1
            Gr_C_dummy_vec = _Gr_C_dummy.flatten().unsqueeze(1) # Make it a column vector (L2*..*LN * 1) or (1*1, 1) if N=1
            
            if Gr_C_dummy_vec.shape[0] == 0: # Handle case where ranks were 0, Gr_C_dummy is empty
                 t_r = torch.zeros((I1_samples, 1), device=device, dtype=X.dtype)
            else:
                 # t_r = X_proj_tr_mat @ torch.linalg.pinv(Gr_C_dummy_vec.T) # This was the previous formula
                 # The HOPLS paper formula uses (_Gr_C_dummy)_vec's pseudo-inverse
                 # This requires careful alignment of indices. Let's re-check the HOPLS paper's formula.
                 # In the HOPLS2 algorithm, t_r is derived from (E_r x_2 P... x_N P)_(1) and vec(G_r^C).
                 # Let Y_tilde = E_r x_2 P... x_N P. Matrizes Y_tilde_(1) (I1 x prod(L)).
                 # G_r^C is (L2..LN, 1). vec(G_r^C) is (prod(L), 1) vector.
                 # The formula in the paper is t_r = Y_tilde_(1) @ vec(G_r^C)^+
                 # Y_tilde_(1) is (I1 x prod(L)). vec(G_r^C) is (prod(L) x 1). vec(G_r^C)^+ is (1 x prod(L)).
                 # t_r = (I1 x prod(L)) @ (1 x prod(L))? No, that's outer product.
                 # It should be t_r is a vector (I1 x 1).
                 # Let's look at the objective: min || Y_tilde - t G^C_{(1)} || where G^C is (prod(L), 1)
                 # Y_tilde is I1 x prod(L). t is I1 x 1. G^C_{(1)} is (1, prod(L)).
                 # Y_tilde = t * G^C_{(1)} (outer product row vector t and row vector G^C_{(1)})
                 # Objective: min || Y_tilde - t g_vec_T ||_F^2 where g_vec_T is (1, prod(L))
                 # LS Solution for t: t = Y_tilde @ g_vec / (g_vec.T @ g_vec)
                 # g_vec is vec(G_r^C) (prod(L) x 1).
                 # So t_r = X_proj_tr_mat @ Gr_C_dummy_vec / torch.norm(Gr_C_dummy_vec)**2

                 Gr_C_dummy_vec_norm_sq = torch.norm(Gr_C_dummy_vec)**2

                 if Gr_C_dummy_vec_norm_sq.abs() < self.epsilon**2:
                      # Avoid division by zero if RHOOI core is near zero
                      t_r = torch.zeros((I1_samples, 1), device=device, dtype=X.dtype)
                 else:
                      t_r = X_proj_tr_mat @ Gr_C_dummy_vec / Gr_C_dummy_vec_norm_sq

            t_r_norm = torch.norm(t_r).clamp(min=self.epsilon)
            t_r = t_r / t_r_norm # Normalise t_r

            t_r_list_accum.append(t_r)
            q_r_list_accum.append(q_r)
            P_r_list_of_lists_accum.append(P_r_current_comp)

            # Calculate G_r using LS formula with the RHOOI-Orth loadings and t_r
            G_r_LS = mode_dot(_X_proj_for_tr, t_r.T, mode=0) # This is correct: (I1, L2..LN) x_0 (I1, 1)^T -> (1, L2..LN)
            
            # Construct MILR weights for G_r
            # MILR weights depend on the shape of G_r and the ranks of the loading matrices P(1)..P(N-1)
            # The core G_r has shape (1, L2, L3, ..., LN)
            # The ranks for the feature modes of G_r are L2, L3, ..., LN
            # These correspond to the factors P(1), P(2), ..., P(N-1) respectively.
            # The number of loading matrices for G_r is N-1.
            milr_ranks_G = self.actual_Ln_used # [L2, L3, ..., LN]
            num_loading_matrices_G = self.N_modal_X # N-1
            W_G_r = _construct_milr_weight_tensor(G_r_LS.shape, milr_ranks_G, self.alpha, num_loading_matrices_G, device)
            
            # Apply MILR penalty
            G_r = G_r_LS / (1.0 + self.lambda_X * W_G_r).clamp(min=self.epsilon)
            G_r_list_accum.append(G_r)

            u_r_vec = Fr_mat @ q_r # (I1, M) @ (M, 1) -> (I1, 1)
            d_r_LS = (t_r.T @ u_r_vec).squeeze() # (1, I1) @ (I1, 1) -> scalar
            
            # Apply standard Ridge penalty to the scalar d_r
            d_r = d_r_LS / (1.0 + self.lambda_Y) # Lambda_Y applies to d_r (the core of Y)
            d_r_list_accum.append(d_r.reshape(1)) # Store as (1,) tensor

            G_r_mat = unfold(G_r, mode=0) # Shape (1, L2*...*LN)
            
            if P_r_current_comp: # If X has feature modes (N > 1)
                P_kron = kronecker([p for p in P_r_current_comp[::-1]]) # Kronecker product P(N-1) (x) ... (x) P(1)
                # Shape (I2*...*IN, L2*...*LN)
                
                norm_G_r_mat_sq = torch.norm(G_r_mat)**2
                if norm_G_r_mat_sq.abs() < self.epsilon**2:
                     G_r_mat_pinv = torch.linalg.pinv(G_r_mat.T) # (L2*..*LN, 1)
                else:
                     G_r_mat_pinv = G_r_mat.T / norm_G_r_mat_sq # (L2*..*LN, 1)
                     
                W_r_val = P_kron @ G_r_mat_pinv # (I2*..*IN, L2*..*LN) @ (L2*..*LN, 1) -> (I2*..*IN, 1)
            else: # X is a vector (N=1), G_r is scalar (1,)
                g_val = G_r.item()
                # If G_r is scalar, X = t*g. t = X/g. W_r should be 1/g.
                # X_mat (I1, 1), T_mat (I1, R). t_r (I1, 1). X_mat @ W_mat = T_mat => (I1, 1) @ (1, R) = (I1, R)
                # W_mat (1, R). For r-th component, X_mat @ W_r = t_r.
                # If X is vector (I1,), X_mat is (I1, 1). W_r should be (1, 1).
                # W_r = 1/g_r where X = t_r * g_r
                if abs(g_val) < self.epsilon:
                    W_r_val = torch.zeros((1, 1), device=device, dtype=X.dtype) # Avoid div by zero
                else:
                    W_r_val = torch.tensor([[1.0 / g_val]], device=device, dtype=X.dtype) # (1, 1) tensor

            W_r_list_accum.append(W_r_val)

            # Deflation using MILR cores and RHOOI-Orth loadings
            Er = Er - tucker_to_tensor((G_r, [t_r] + P_r_current_comp)) # G_r (1, L2..LN), t_r (I1, 1), P_r (Ii, Li)
            Fr_mat = Fr_mat - d_r * (t_r @ q_r.T) # d_r scalar, t_r (I1, 1), q_r (M, 1) -> (I1, M)

        self.num_components_fitted = num_fitted_for_loop
        self.P_r_all_components = P_r_list_of_lists_accum
        self.Q_r_all_components = torch.cat(q_r_list_accum, dim=1) if q_r_list_accum else torch.empty((M_responses_Y, 0), device=device, dtype=X.dtype)
        self.G_r_all_components = G_r_list_accum
        self.D_r_all_components = torch.cat(d_r_list_accum) if d_r_list_accum else torch.empty(0, device=device, dtype=X.dtype)
        self.T_mat = torch.cat(t_r_list_accum, dim=1) if t_r_list_accum else torch.empty((I1_samples, 0), device=device, dtype=X.dtype)
        
        # Determine the size of the feature dimension for W_mat
        # This should be the product of the sizes of X's non-sample modes.
        if X.ndim == 1: # X is (N_samples,)
             prod_X_non_sample_dims = 1
        else: # X is (N_samples, I2, ..., IN)
             prod_X_non_sample_dims = X.shape[1:].numel()

        self.W_mat = torch.cat(W_r_list_accum, dim=1) if W_r_list_accum else torch.empty((prod_X_non_sample_dims, 0), device=device, dtype=X.dtype)
        
        return self

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> "HOPLS_MILR_RHOOI":
        X = X.to(device)
        Y = Y.to(device)

        # Ensure Y is at least a matrix (N_samples x M)
        if Y.ndim <= 1: 
            Y = Y.unsqueeze(1) # Make it (N_samples, 1) if Y was vector or scalar
        
        # If Y is a matrix (N_samples x M), use the Tensor X, Matrix Y logic
        if Y.ndim == 2:
            return self._fit_tensor_X_matrix_Y(X, Y)
        
        # Otherwise, Y is a tensor (N_samples x J2 x ... x JM)
        self._is_matrix_Y = False
        I1_samples = X.shape[0]
        
        self.N_modal_X = X.ndim - 1
        self.M_modal_Y = Y.ndim - 1
        
        self.actual_Ln_used = list(self.Ln)
        # For Tensor Y, Km must be provided and match Y.ndim - 1
        if self.Km is None:
             raise ValueError("Km (list of ranks for Y feature modes) must be provided when Y is a tensor.")
        self.actual_Km_used = list(self.Km) 

        if self.N_modal_X > 0 and len(self.actual_Ln_used) != self.N_modal_X:
            raise ValueError(f"Length of Ln ({len(self.actual_Ln_used)}) must match X.ndim-1 ({self.N_modal_X}) for tensor X")
        if self.M_modal_Y > 0 and len(self.actual_Km_used) != self.M_modal_Y:
             raise ValueError(f"Length of Km ({len(self.actual_Km_used)}) must match Y.ndim-1 ({self.M_modal_Y}) for tensor Y")
        
        # Handle edge case: X vector (N=1), Y tensor (M>1). N_modal_X = 0, M_modal_Y > 0. Ln=[], Km=[K2..KM].
        if self.N_modal_X == 0: self.actual_Ln_used = []
        if self.M_modal_Y == 0: self.actual_Km_used = [] # Should not happen if Y.ndim > 2, but defensive

        Er, Fr = X.clone(), Y.clone()

        P_r_list_of_lists_accum: List[List[torch.Tensor]] = []
        Q_r_list_of_lists_accum: List[List[torch.Tensor]] = []
        G_r_list_accum: List[torch.Tensor] = []
        D_r_list_accum: List[torch.Tensor] = []
        t_r_list_accum: List[torch.Tensor] = []
        W_r_list_accum: List[torch.Tensor] = []

        num_fitted_for_loop = 0
        for _r_component_idx in range(self.R):
            if torch.norm(Er) < self.epsilon or torch.norm(Fr) < self.epsilon:
                if self.rhooi_verbose:
                    print(f"Stopping at component {_r_component_idx + 1} due to small residual norm.")
                break
            num_fitted_for_loop += 1
            
            # Cr = <Er, Fr>_{1;1} -> Tensor contraction over mode 0 (samples)
            # Shape of Cr: (I2..IN, J2..JM)
            Cr = torch.tensordot(Er, Fr, dims=([0], [0])) 
            
            # Cr modes: 0..N-2 correspond to X modes 2..N -> P(1)..P(N-1)
            # Cr modes N-1..N-1+M-1-1 correspond to Y modes 2..M -> Q(1)..Q(M-1)
            
            ranks_for_Cr_RHOOI = self.actual_Ln_used + self.actual_Km_used
            penalties_for_Cr_RHOOI = [self.lambda_P_factor_penalty] * self.N_modal_X + [self.lambda_Q_factor_penalty] 
            

            # RHOOI on Cr (I2..IN, J2..JM) yields P(1)..P(N-1) and Q(1)..Q(M-1) and core_C
            
            if Cr.ndim == 0: # X vector, Y vector case 

                 if self.rhooi_verbose: print("Error: Reached unexpected scalar Cr state in Tensor-Tensor path.")
                 break # Should not happen
            
            _Gr_C_dummy, factors_Cr = _rhooi_orth(
                Cr, ranks_for_Cr_RHOOI, penalties_for_Cr_RHOOI,
                init="svd", n_iter_max=self.rhooi_n_iter_max, tol=self.rhooi_tol,
                verbose=self.rhooi_verbose, dev=device
            )

            # Factors from RHOOI on Cr (I2..IN, J2..JM):
            # factors_Cr[0]..factors_Cr[N_modal_X-1] are P(1)..P(N-1)
            # factors_Cr[N_modal_X]..factors_Cr[N_modal_X+M_modal_Y-1] are Q(1)..Q(M-1)
            P_r_current_comp = factors_Cr[:self.N_modal_X]
            Q_r_current_comp = factors_Cr[self.N_modal_X:]
            
            # RHOOI-Orth should already return orthonormal factors. No extra norm needed.

            # Step 2.3 from original HOPLS (Tensor-Tensor): Find t_r from SVD of projected X
            _X_proj_for_tr = Er
            if P_r_current_comp: # Apply P factors if they exist (N > 1)
                _X_proj_for_tr = multi_mode_dot(Er, [p.T for p in P_r_current_comp], modes=list(range(1, self.N_modal_X + 1)))
            
            U_tr, _, _ = torch.linalg.svd(unfold(_X_proj_for_tr, mode=0), full_matrices=False)
            t_r = U_tr[:, 0:1]
            t_r_norm = torch.norm(t_r).clamp(min=self.epsilon)
            t_r = t_r / t_r_norm # Normalise t_r

            t_r_list_accum.append(t_r)
            P_r_list_of_lists_accum.append(P_r_current_comp)
            Q_r_list_of_lists_accum.append(Q_r_current_comp)

            # Calculate G_r using LS formula with the RHOOI-Orth loadings and t_r
            G_r_LS = mode_dot(_X_proj_for_tr, t_r.T, mode=0) # Shape (1, L2..LN)
            
            # Construct MILR weights for G_r
            milr_ranks_G = self.actual_Ln_used # [L2, L3, ..., LN]
            num_loading_matrices_G = self.N_modal_X # N-1
            W_G_r = _construct_milr_weight_tensor(G_r_LS.shape, milr_ranks_G, self.alpha, num_loading_matrices_G, device)
            
            # Apply MILR penalty
            G_r = G_r_LS / (1.0 + self.lambda_X * W_G_r).clamp(min=self.epsilon)
            G_r_list_accum.append(G_r)

            # Calculate D_r using LS formula with RHOOI-Orth loadings and t_r
            _Y_proj_for_Dr = Fr
            if Q_r_current_comp: # Apply Q factors if they exist (M > 1)
                _Y_proj_for_Dr = multi_mode_dot(Fr, [q.T for q in Q_r_current_comp], modes=list(range(1, self.M_modal_Y + 1)))
            D_r_LS = mode_dot(_Y_proj_for_Dr, t_r.T, mode=0) # Shape (1, K2..KM)
            
            # Construct MILR weights for D_r
            milr_ranks_D = self.actual_Km_used # [K2, ..., KM]
            num_loading_matrices_D = self.M_modal_Y # M-1
            W_D_r = _construct_milr_weight_tensor(D_r_LS.shape, milr_ranks_D, self.alpha, num_loading_matrices_D, device)

            # Apply MILR penalty
            D_r = D_r_LS / (1.0 + self.lambda_Y * W_D_r).clamp(min=self.epsilon)
            D_r_list_accum.append(D_r)
            
            # Calculate W_r for prediction - Same logic as Tensor-Matrix case
            G_r_mat = unfold(G_r, mode=0) # Shape (1, L2*...*LN)
            
            if P_r_current_comp: # If X has feature modes (N > 1)
                P_kron = kronecker([p for p in P_r_current_comp[::-1]]) # P(N-1) (x) ... (x) P(1)
                # Shape (I2*...*IN, L2*...*LN)
                
                norm_G_r_mat_sq = torch.norm(G_r_mat)**2
                if norm_G_r_mat_sq.abs() < self.epsilon**2:
                     G_r_mat_pinv = torch.linalg.pinv(G_r_mat.T) # (L2*..*LN, 1)
                else:
                     G_r_mat_pinv = G_r_mat.T / norm_G_r_mat_sq # (L2*..*LN, 1)
                     
                W_r_val = P_kron @ G_r_mat_pinv # (I2*..*IN, 1)
            else: # X is a vector (N=1), G_r is scalar (1,)
                g_val = G_r.item()
                if abs(g_val) < self.epsilon:
                    W_r_val = torch.zeros((1, 1), device=device, dtype=X.dtype)
                else:
                    W_r_val = torch.tensor([[1.0 / g_val]], device=device, dtype=X.dtype) # (1, 1) tensor

            W_r_list_accum.append(W_r_val)

            # Deflation using MILR cores and RHOOI-Orth loadings
            Er = Er - tucker_to_tensor((G_r, [t_r] + P_r_current_comp)) # G_r (1, L2..LN), t_r (I1, 1), P_r (Ii, Li)
            Fr = Fr - tucker_to_tensor((D_r, [t_r] + Q_r_current_comp)) # D_r (1, K2..KM), t_r (I1, 1), Q_r (Ji, Ki)

        self.num_components_fitted = num_fitted_for_loop
        self.P_r_all_components = P_r_list_of_lists_accum
        self.Q_r_all_components = Q_r_list_of_lists_accum # List of lists for Tensor Y
        self.G_r_all_components = G_r_list_accum
        self.D_r_all_components = D_r_list_accum # List of tensors for Tensor Y
        self.T_mat = torch.cat(t_r_list_accum, dim=1) if t_r_list_accum else torch.empty((I1_samples, 0), device=device, dtype=X.dtype)
        
        # Determine the size of the feature dimension for W_mat
        if X.ndim == 1: # X is (N_samples,)
             prod_X_non_sample_dims = 1
        else: # X is (N_samples, I2, ..., IN)
             prod_X_non_sample_dims = X.shape[1:].numel()

        self.W_mat = torch.cat(W_r_list_accum, dim=1) if W_r_list_accum else torch.empty((prod_X_non_sample_dims, 0), device=device, dtype=X.dtype)

        return self
    
    def predict(
        self, X_new: torch.Tensor, Y_true_for_shape_and_metric: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, int, List[float]]:
        if self.T_mat is None or self.W_mat is None or self.num_components_fitted == 0:
             # If num_components_fitted is 0, model wasn't fit or residuals were too small initially.
             # Return zeros with a determined shape.
             output_Y_shape = list(X_new.shape[:1])
             if Y_true_for_shape_and_metric is not None:
                 output_Y_shape.extend(Y_true_for_shape_and_metric.shape[1:])
             elif self._is_matrix_Y:
                 # For Matrix Y, shape is (N_samples, M_responses_Y). Need M_responses_Y from fit.
                 # Can try to infer from Q_r_all_components if it's a tensor (Tensor-Matrix case)
                 if isinstance(self.Q_r_all_components, torch.Tensor) and self.Q_r_all_components.shape[0] > 0:
                      output_Y_shape.append(self.Q_r_all_components.shape[0])
                 else: # Fallback if no Q or M dim not stored explicitly
                      output_Y_shape.append(1) # Assume 1 output feature
             else: # Tensor Y
                 # For Tensor Y, shape is (N_samples, J2..JM). Need J2..JM from fit.
                 # Can try to infer from D_r_all_components if it's a list of tensors.
                 if isinstance(self.D_r_all_components, list) and self.D_r_all_components:
                     output_Y_shape.extend(self.D_r_all_components[0].shape[1:])
                 else: # Fallback if no D or shape not stored
                     output_Y_shape.append(1) # Assume 1 output feature (makes Y (N_samples, 1))

             # Ensure Y_pred is at least 2D (N_samples x 1)
             if len(output_Y_shape) == 1 and X_new.shape[0] > 0:
                  output_Y_shape.append(1)
             elif len(output_Y_shape) == 0 and X_new.shape[0] > 0: # If X_new was scalar? (not handled currently)
                  output_Y_shape = [X_new.shape[0], 1]
             elif len(output_Y_shape) == 0 and X_new.shape[0] == 0: # Empty input
                  output_Y_shape = [0, 0] # Or [0, 1] depending on convention

             return torch.zeros(output_Y_shape, device=device, dtype=X_new.dtype), 0, []

        X_new_dev = X_new.to(device)
        # Handle if X_new is a vector (N_samples,)
        if X_new_dev.ndim == 1: 
            X_new_mat = X_new_dev.unsqueeze(1) # Treat as N_samples x 1
        else:
            X_new_mat = unfold(X_new_dev, mode=0) # Matrize (N_new_samples, prod(I2..IN))

        # Ensure W_mat feature dim matches X_new_mat feature dim
        if self.W_mat.shape[0] != X_new_mat.shape[1]:
             raise ValueError(f"W_mat feature dim ({self.W_mat.shape[0]}) != X_new feature dim ({X_new_mat.shape[1]}). "
                              "Input X_new shape may be inconsistent with training X shape.")


        T_new_all_r_mat = X_new_mat @ self.W_mat # (N_new_samples, prod(I2..IN)) @ (prod(I2..IN), R_fitted) -> (N_new_samples, R_fitted)

        best_q2_val = -float('inf')
        best_Y_pred_val: Optional[torch.Tensor] = None
        best_r_val = 0
        q2s_list: List[float] = []

        Y_true_dev: Optional[torch.Tensor] = None
        if Y_true_for_shape_and_metric is not None:
            Y_true_dev = Y_true_for_shape_and_metric.to(device)
            # Ensure Y_true_dev is at least 2D for metric calculation
            if Y_true_dev.ndim <=1: Y_true_dev = Y_true_dev.unsqueeze(1) if Y_true_dev.ndim == 1 else Y_true_dev.reshape(Y_true_dev.shape[0],1)
            Y_true_is_provided = True
            output_Y_shape = list(Y_true_dev.shape) # Use Y_true shape as the target shape
        else:
            Y_true_is_provided = False
            # Determine the target shape for Y_pred based on model type and stored info
            output_Y_shape = list(X_new_dev.shape[:1]) # Start with sample size
            if self._is_matrix_Y:
                 if isinstance(self.Q_r_all_components, torch.Tensor):
                     output_Y_shape.append(self.Q_r_all_components.shape[0]) # (N_samples, M)
                 else: # Should not happen if model fit Matrix Y, but defensive
                      output_Y_shape.append(1)
            else: # Tensor Y
                 if isinstance(self.D_r_all_components, list) and self.D_r_all_components:
                      output_Y_shape.extend(self.D_r_all_components[0].shape[1:]) # (N_samples, J2..JM)
                 else: # Should not happen if model fit Tensor Y, but defensive
                      output_Y_shape.append(1) # Fallback to (N_samples, 1)

            # Ensure Y_pred is at least 2D (N_samples x 1)
            if len(output_Y_shape) == 1 and X_new_dev.shape[0] > 0:
                 output_Y_shape.append(1)


        for r_to_use in range(1, self.num_components_fitted + 1):
            T_new_current_r_iter = T_new_all_r_mat[:, :r_to_use] 

            if self._is_matrix_Y:
                # Q_r_all_components is (M, R_fitted) tensor
                Q_r_subset = self.Q_r_all_components[:, :r_to_use] 
                # D_r_all_components is (R_fitted,) tensor
                D_r_diag_subset = torch.diag(self.D_r_all_components[:r_to_use]) 
                # Y_pred = T @ D_diag @ Q^T
                # (N_new_samples, r_to_use) @ (r_to_use, r_to_use) @ (M, r_to_use)^T -> (N_new_samples, M)
                Y_pred_current_iter = T_new_current_r_iter @ D_r_diag_subset @ Q_r_subset.T
                
            else: # Tensor Y
                Y_pred_current_iter = torch.zeros(output_Y_shape, device=device, dtype=X_new_dev.dtype)
                for i_comp_idx in range(r_to_use):
                    # t_new_for_comp is (N_new_samples, 1)
                    t_new_for_comp = T_new_current_r_iter[:, i_comp_idx:i_comp_idx+1]
                    # D_comp is the MILR core for the i_comp_idx component (1, K2..KM)
                    D_comp = self.D_r_all_components[i_comp_idx]            
                    # Q_comp_list is the list of loading matrices for the i_comp_idx component's Y-side ([Q(1)..Q(M-1)])
                    Q_comp_list = self.Q_r_all_components[i_comp_idx]       
                    
                    # Reconstruct Y component: D_comp x_0 t_new x_1 Q(1) ... x_M-1 Q(M-1)
                    # This is tucker_to_tensor((D_comp, [t_new_for_comp] + Q_comp_list))
                    term_Y_pred = tucker_to_tensor((D_comp, [t_new_for_comp] + Q_comp_list))
                    Y_pred_current_iter = Y_pred_current_iter + term_Y_pred
            
            # --- Metric Calculation ---
            if Y_true_is_provided and Y_pred_current_iter.numel() > 0 and Y_true_dev.numel() > 0 :
                
                # Reshape Y_pred to match Y_true if necessary (robustness for metric)
                try:
                    Y_pred_reshaped = Y_pred_current_iter.reshape_as(Y_true_dev)
                except RuntimeError:
                     # If reshaping fails, it's a shape mismatch we can't recover from
                     q2_val = -float('inf') # Penalize shape mismatch severely
                     if self.rhooi_verbose: print(f"Shape mismatch for Q2 calculation at r={r_to_use}: Y_pred_current_iter.shape={Y_pred_current_iter.shape}, Y_true_dev.shape={Y_true_dev.shape}")
                     
                else: # Reshape was successful
                    q2_val = self.metric(Y_true_dev, Y_pred_reshaped)

                q2s_list.append(q2_val)

                # Track best component based on Q2
                if q2_val > best_q2_val:
                    best_q2_val = q2_val
                    best_Y_pred_val = Y_pred_current_iter # Store the prediction in its original shape
                    best_r_val = r_to_use

            # --- No Y_true provided: return prediction for max R ---
            elif r_to_use == self.num_components_fitted and not Y_true_is_provided:
                best_Y_pred_val = Y_pred_current_iter 
                best_r_val = r_to_use

        # --- Final Return Value ---
        if best_Y_pred_val is None: 
            # This might happen if num_components_fitted > 0 but all Q2 were -inf
            # In this case, return the prediction for the last component fitted, or zeros if no components fit
            if self.num_components_fitted > 0 and 'Y_pred_current_iter' in locals() and Y_pred_current_iter.numel() > 0:
                 best_Y_pred_val = Y_pred_current_iter # Y_pred_current_iter will be the last one calculated (for R_fitted)
                 best_r_val = self.num_components_fitted
            else:
                 # This case should ideally not be reached if num_components_fitted == 0 check works,
                 # but provides a safe fallback.
                 best_Y_pred_val = torch.zeros(output_Y_shape, device=device, dtype=X_new_dev.dtype)
                 best_r_val = 0
                 
        return best_Y_pred_val, best_r_val, q2s_list

    def score(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        # Fit if not already fitted (score implicitly fits)
        if self.T_mat is None or self.num_components_fitted == 0: 
             try:
                 self.fit(X,Y)
             except Exception as e:
                  print(f"Error during implicit fit in score method: {e}")
                  return -float('inf') # Indicate failure to fit


        # Predict using X and evaluate against Y
        # The predict method calculates Q2 for each component and returns the max if Y_true is provided
        _Y_pred, _best_r, q2s = self.predict(X, Y_true_for_shape_and_metric=Y)
        
        if not q2s:
            return -float('inf') # No components fit or Q2 calculation failed
            
        return max(q2s)

