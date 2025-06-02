# Uses Ridge solution for Low-Rank Approx for Tensor-Matrix case. This is unfortunately N-PLS, not HOPLS.
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorly as tl
import torch
from tensorly import tucker_to_tensor, fold
from tensorly.decomposition import tucker
from tensorly.tenalg import kronecker, mode_dot, multi_mode_dot

# Use PyTorch backend throughout
tl.set_backend("pytorch")
# Automatically use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set default dtype
torch.set_default_dtype(torch.float64)


def matricize(data: torch.Tensor) -> torch.Tensor:
    """
    Unfold a tensor into a matrix along mode-1 (axis 0),
    using Fortran order so that fibres are contiguous.
    E.g. a 4×3×2 tensor → 4×6 matrix.
    """
    # NEW: Pure-PyTorch unfold (no CPU<->NumPy copy), using permute+reshape
    # Move tensor to contiguous layout on correct device
    data = data.contiguous()
    # Flatten all but first dim
    return data.view(data.shape[0], -1)

def matricize_n(data: torch.Tensor, mode: int = 0) -> torch.Tensor:
    """
    Unfold a tensor into a matrix along a specified mode.
    tensorly.unfold(data, mode) uses C-order by default.
    To match typical Fortran-order unfolding (like MATLAB's reshape or your original matricize):
    matricize(X) (mode 0) -> X.reshape(X.shape[0], -1)
    """
    if mode == 0:
        return data.contiguous().view(data.shape[0], -1)
    else:
        # For other modes, tensorly.unfold is fine, or a permute+reshape
        return tl.unfold(data, mode)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Square Error between two arrays."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0)))

def qsquared(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Compute the Q² statistic (1 – PRESS/TSS) for two tensors/matrices.
    Used as the default metric in HOPLS.
    """
    return float(1 - (torch.norm(y_true - y_pred) ** 2) / (torch.norm(y_true) ** 2))


def ridge_pinv(A: torch.Tensor, lam: float) -> torch.Tensor:
    """(AᵀA+λI)⁻¹Aᵀ   – works on GPU, broadcasts batch dims if present."""
    if lam == 0.0:
        return torch.linalg.pinv(A)
    AtA = A.T @ A
    n = AtA.shape[-1]
    return torch.linalg.solve(
        AtA + lam * torch.eye(n, device=A.device, dtype=A.dtype), A.T
    )


def qsquared(Y, Yhat):
    """Coefficient of prediction Q²."""
    sse = torch.sum((Y - Yhat) ** 2)
    sst = torch.sum((Y - torch.mean(Y, dim=0, keepdim=True)) ** 2)
    return 1.0 - sse / sst


class HOPLS_RIDGE:
    """
    HOPLS with a global ℓ₂ (ridge) penalty applied to all least-squares solves.
    """

    def __init__(
        self,
        R: int,
        Ln: List[int],
        Km: Optional[List[int]] = None,
        metric: Optional[callable] = None,
        epsilon: float = 1e-6,
        ridge: float = 1e-3,
    ) -> None:
        self.R = R
        self.Ln = Ln
        self.Km = Km if Km is not None else [Ln[0]]
        self.metric = metric or qsquared
        self.epsilon = epsilon
        self.lam = float(ridge)

        self.N = len(self.Ln)  # non-sample modes of X
        self.M = len(self.Km)  # non-sample modes of Y
        self._is_2d = False  # flag set in fit()
        self.model: Optional[Tuple] = None

    # -----------------------------------------------------------------
    #  Algorithm 2: special case Y ∈ ℝⁿˣᵐ (matrix-response)
    # -----------------------------------------------------------------
    def _fit_2d(self, X: torch.Tensor, Y: torch.Tensor) -> "HOPLS_RIDGE":
        Er, Fr = X.clone().to(device), Y.clone().to(device)

        P: List[List[torch.Tensor]] = []
        Q: List[torch.Tensor] = []
        d_list: List[float] = []
        T: List[torch.Tensor] = []
        W: List[torch.Tensor] = []

        for r in range(self.R):
            if torch.norm(Er) <= self.epsilon or torch.norm(Fr) <= self.epsilon:
                break

            # 1) cross-covariance tensor
            Cr = mode_dot(Er, Fr.T, mode=0)

            # 2) Tucker rank-[1]+Ln → (GrC, [q_r, P_r¹…P_rᴺ])
            GrC, factors = tucker(Cr, rank=[1] + self.Ln, init="svd")
            q_r = factors[0] / torch.norm(factors[0])
            P_r = [p / torch.norm(p) for p in factors[1:]]

            # 3) latent vector  t_r = mat(t̃) · pinv(mat(GrC))
            t_tilde = multi_mode_dot(
                Er, P_r, modes=list(range(1, len(P_r) + 1)), transpose=True
            )
            GrC_pinv = ridge_pinv(matricize(GrC), self.lam)  # (k × 1)
            t_r = matricize(t_tilde) @ GrC_pinv  # (I₁ × 1)
            t_r = t_r / torch.norm(t_r)

            # 4) scalar regression with ridge: d_r = (uᵀ t)/(tᵀ t + λ)
            u_r = Fr @ q_r
            num = (u_r.T @ t_r).item()
            den = (t_r.T @ t_r).item() + self.lam
            d_r = num / den
            d_list.append(d_r)

            # 5) X-weights: W_r = kron(P_r) · pinv(mat(G_d))
            G_d = multi_mode_dot(
                Er, [t_r.T] + [p.T for p in P_r], modes=[0] + list(range(1, self.N + 1))
            )
            W_r = kronecker(P_r[::-1]) @ ridge_pinv(matricize(G_d), self.lam)

            # 6) deflation
            Er = Er - multi_mode_dot(
                G_d, [t_r] + P_r, modes=[0] + list(range(1, self.N + 1))
            )
            Fr = Fr - d_r * (t_r @ q_r.T)

            # collect
            P.append(P_r)
            Q.append(q_r)
            T.append(t_r)
            W.append(W_r)

        # stack heavy blocks
        T_mat = torch.cat(T, dim=1)  # I₁×r
        W_mat = torch.cat(W, dim=1)  # p×r  (p = prod of X-modes)

        # save model
        self._is_2d = True
        self.model = (P, Q, d_list, T_mat, W_mat)
        return self

    # -----------------------------------------------------------------
    #  Algorithm 1: full tensor-response
    # -----------------------------------------------------------------
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> "HOPLS_RIDGE":
        assert X.ndim >= 3 and Y.ndim >= 2
        if Y.ndim == 2:
            return self._fit_2d(X, Y)

        assert len(self.Ln) == X.ndim - 1
        assert len(self.Km) == Y.ndim - 1

        Er, Fr = X.clone().to(device), Y.clone().to(device)

        P_all: List[List[torch.Tensor]] = []
        Q_all: List[List[torch.Tensor]] = []
        D_all: List[torch.Tensor] = []
        T: List[torch.Tensor] = []
        W: List[torch.Tensor] = []

        for r in range(self.R):
            if torch.norm(Er) <= self.epsilon or torch.norm(Fr) <= self.epsilon:
                break

            # 1) mode-1 cross-covariance
            Cr = torch.tensordot(Er, Fr, dims=([0], [0]))

            # 2) joint Tucker (Ln+Km)
            _, factors = tucker(Cr, rank=self.Ln + self.Km, init="svd")
            Pr, Qr = factors[: self.N], factors[self.N :]

            # 3) latent vector t_r via SVD (could be ridge-extended)
            Er_proj = multi_mode_dot(
                Er, Pr, modes=list(range(1, self.N + 1)), transpose=True
            )
            U, _, _ = torch.linalg.svd(matricize(Er_proj))
            t_r = U[:, :1]

            # 4) core tensors G_r, D_r
            G_r = multi_mode_dot(
                Er, [t_r.T] + [p.T for p in Pr], modes=[0] + list(range(1, self.N + 1))
            )
            D_r = multi_mode_dot(
                Fr, [t_r.T] + [q.T for q in Qr], modes=[0] + list(range(1, self.M + 1))
            )
            D_all.append(D_r)

            # 5) X-weights with ridge
            W_r = kronecker(Pr[::-1]) @ ridge_pinv(matricize(G_r), self.lam)

            # 6) deflation
            Er = Er - multi_mode_dot(
                G_r, [t_r] + Pr, modes=[0] + list(range(1, self.N + 1))
            )
            Fr = Fr - multi_mode_dot(
                D_r, [t_r] + Qr, modes=[0] + list(range(1, self.M + 1))
            )

            # collect
            P_all.append(Pr)
            Q_all.append(Qr)
            T.append(t_r)
            W.append(W_r)

        # final stacks
        T_mat = torch.cat(T, dim=1)
        W_mat = torch.cat(W, dim=1)

        self._is_2d = False
        self.model = (P_all, Q_all, D_all, T_mat, W_mat)
        return self

    # -----------------------------------------------------------------
    #  prediction
    # -----------------------------------------------------------------
    def predict(
        self, X: torch.Tensor, Y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, int, List[float]]:
        """
        Returns:
            Y_pred : Tensor of shape matching Y (for 2d) or X-mode-0 shape + Y-modes
            best_r : int (always self.R here)
            q2s    : empty list (Q² not computed in ridge version)
        """
        assert self.model is not None, "Model not fitted."
        X_mat = matricize(X).to(device)

        if self._is_2d:
            P, Q, d_list, _, W = self.model  # unpack 2d branch
            Q_mat = torch.cat(Q, dim=1)  # m×r
            D = torch.diag(torch.tensor(d_list, device=device, dtype=X.dtype))  # r×r
            # Ŷ = X·(W·D·Qᵀ)
            Y_pred = X_mat @ (W @ D @ Q_mat.T)
            # reshape back to (n_samples, m)
            return Y_pred, self.R, []

        # tensor-response branch
        P_all, Q_all, D_all, _, W = self.model

        # build Q_star = [ D_r(1st-mode fold) · kron(Q_r factors) ]  stacked
        Q_star_parts = []
        for r, D_r in enumerate(D_all):
            Qkron = kronecker([Q_all[r][self.M - m - 1] for m in range(self.M)])
            Q_star_parts.append(matricize(D_r) @ Qkron.T)
        Q_star = torch.cat(Q_star_parts, dim=0)  # (r, ∏Y-dims)

        # full-rank prediction
        Z = X_mat @ (W @ Q_star)
        Y_pred = fold(
            Z, mode=0, shape=X.shape[:1] + tuple(D_r.shape[1:] for D_r in D_all[:1])[0]
        )

        return Y_pred, self.R, []

    # -----------------------------------------------------------------
    def score(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        self.fit(X, Y)
        Y_pred, _, _ = self.predict(X, Y)
        return self.metric(
            Y.reshape(Y.shape[0], -1), Y_pred.cpu().reshape(Y_pred.shape[0], -1)
        )

def _construct_milr_weight_tensor(
    core_shape: Tuple[int, ...],
    mode_ranks_for_core_definition: List[int], # Ranks [L2, L3,...,LN] or [K2, K3,...,KM]
    alpha: float,
    num_loading_matrices_for_core: int, # N-1 for G_r, M-1 for D_r
    core_tensor_device: torch.device
) -> torch.Tensor:
    """
    Constructs the mean index-weighted L2 regularization weight tensor.
    core_shape: (1, L_mode1, L_mode2, ..., L_mode_NumLoadings) e.g., (1, L2, L3..LN) for G_r
    mode_ranks_for_core_definition: [L_mode1, L_mode2, ..., L_mode_NumLoadings]
                                     e.g. self.Ln = [L2,L3..LN] for G_r
    num_loading_matrices_for_core: N-1 for G_r (number of P matrices)
    """
    if num_loading_matrices_for_core == 0:
        return torch.tensor(0.0, device=core_tensor_device)

    if len(mode_ranks_for_core_definition) != num_loading_matrices_for_core:
        # This can happen if X is a matrix, N-1=1, but self.Ln might be empty if not defined for matrix.
        # Assuming if mode_ranks is empty and num_loading_matrices > 0, it's an issue.
        # If core_shape indicates multiple modes but mode_ranks is short, it's an issue.
        if num_loading_matrices_for_core > 0 and not mode_ranks_for_core_definition:
             raise ValueError("mode_ranks_for_core_definition is empty but num_loading_matrices > 0")


    weight_tensor = torch.zeros(core_shape, device=core_tensor_device)
    
    # Create iterators for indices l2, ..., l_N (0-based for PyTorch)
    # mode_ranks_for_core_definition are the L_k values for these modes
    iter_ranges = [torch.arange(rank, device=core_tensor_device) for rank in mode_ranks_for_core_definition]

    for current_mode_indices_in_core in torch.cartesian_prod(*iter_ranges):
        # current_mode_indices_in_core is (l2_idx, l3_idx, ..., lN_idx) for G_r, 0-based
        sum_norm_indices_alpha = 0.0
        for i in range(num_loading_matrices_for_core):
            # current_mode_indices_in_core[i] is the 0-based index along the (i+1)-th mode of G_r (excluding mode 0)
            # This corresponds to the (i+1)-th loading matrix P_r^(i+1) or Q_r^(i+1)
            # mode_ranks_for_core_definition[i] is its total rank (L_{i+2} or K_{i+2})
            
            one_based_idx = current_mode_indices_in_core[i].item() + 1
            max_rank_for_this_mode = mode_ranks_for_core_definition[i]
            
            if max_rank_for_this_mode > 0:
                 sum_norm_indices_alpha += (one_based_idx / max_rank_for_this_mode) ** alpha
        
        # Construct the full index for assignment, prepending 0 for the first mode (size 1)
        full_slice_index = tuple([0] + current_mode_indices_in_core.tolist())
        weight_tensor[full_slice_index] = (1.0 / num_loading_matrices_for_core) * sum_norm_indices_alpha
        
    return weight_tensor

def qsquared_score(Y_true: torch.Tensor, Y_pred: torch.Tensor) -> float:
    Y_true_dev = Y_true.to(Y_pred.device) # Ensure same device
    sse = torch.sum((Y_true_dev - Y_pred) ** 2)
    sst_val = torch.sum((Y_true_dev - torch.mean(Y_true_dev, dim=0, keepdim=True)) ** 2)
    if sst_val < 1e-9: # Avoid division by zero or near-zero TSS
        return 1.0 if sse < 1e-9 else 0.0 
    return float(1.0 - sse / sst_val)


class HOPLS_MILR:
    def __init__(
        self,
        R: int,
        Ln: List[int],
        Km: Optional[List[int]] = None,
        metric: Optional[callable] = None,
        epsilon: float = 1e-6,
        lambda_X: float = 1e-3,
        lambda_Y: float = 1e-3,
        alpha: float = 1.0,
    ) -> None:
        self.R = R
        self.Ln = Ln # Ranks for X modes [L2, L3, ..., LN] corresponding to P^(1)...P^(N-1)
        self.Km_param = Km # Ranks for Y modes [K2, K3, ..., KM] corresponding to Q^(1)...Q^(M-1)
        self.metric = metric or qsquared_score
        self.epsilon = epsilon
        self.lambda_X = float(lambda_X)
        self.lambda_Y = float(lambda_Y)
        self.alpha = float(alpha)

        self._is_matrix_Y: bool = False
        self.N_modal_X: int = 0 # Number of loading matrices P for X (X.ndim-1)
        self.M_modal_Y: int = 0 # Number of loading matrices Q for Y (Y.ndim-1)
        self.actual_Km: List[int] = [] # Actual Km used, derived in fit()

        # Model components to be stored
        self.P_r_all_components: List[List[torch.Tensor]] = [] # For X: list (over r) of lists (P_r^(j))
        self.Q_r_all_components: Union[List[List[torch.Tensor]], torch.Tensor] = [] # For Y: list of lists (Q_r^(j)) or M x R matrix q_r
        self.G_r_all_components: List[torch.Tensor] = [] # For X: list of G_r
        self.D_r_all_components: Union[List[torch.Tensor], torch.Tensor] = [] # For Y: list of D_r or R vector d_r
        self.T_mat: Optional[torch.Tensor] = None # I1 x R
        self.W_mat: Optional[torch.Tensor] = None # (Prod X_modes_non_sample) x R

    def _fit_tensor_X_matrix_Y(self, X: torch.Tensor, Y: torch.Tensor) -> "HOPLS_MILR":
        self._is_matrix_Y = True
        I1_samples = X.shape[0]
        M_responses = Y.shape[1]
        self.N_modal_X = X.ndim - 1
        
        if len(self.Ln) != self.N_modal_X and self.N_modal_X > 0 :
            raise ValueError(f"Length of Ln ({len(self.Ln)}) must match X.ndim-1 ({self.N_modal_X})")

        Er, Fr_mat = X.clone(), Y.clone()

        P_r_list_of_lists: List[List[torch.Tensor]] = []
        q_r_list: List[torch.Tensor] = []
        G_r_list: List[torch.Tensor] = []
        d_r_list: List[torch.Tensor] = []
        t_r_list: List[torch.Tensor] = []
        W_r_list: List[torch.Tensor] = []

        for r_component in range(self.R):
            if torch.norm(Er) < self.epsilon or torch.norm(Fr_mat) < self.epsilon: # Use < for epsilon check
                break

            Cr = mode_dot(Er, Fr_mat.T, mode=0)
            ranks_for_Cr_HOOI = [1] + self.Ln if self.N_modal_X > 0 else [1]
            Gr_C, factors_Cr = tucker(Cr, rank=ranks_for_Cr_HOOI, init="svd", n_iter_max=15)
            
            q_r = factors_Cr[0] / torch.norm(factors_Cr[0]).clamp(min=1e-9)
            P_r_current = [p / torch.norm(p, dim=0, keepdim=True).clamp(min=1e-9) for p in factors_Cr[1:]]

            _X_proj_for_svd = Er
            if self.N_modal_X > 0:
                _X_proj_for_svd = multi_mode_dot(Er, [p.T for p in P_r_current], modes=list(range(1, self.N_modal_X + 1)))
            
            U_tr, _, _ = torch.linalg.svd(matricize_n(_X_proj_for_svd, mode=0), full_matrices=False)
            t_r = U_tr[:, 0:1]
            if torch.norm(t_r) > 1e-9 : t_r = t_r / torch.norm(t_r) # Normalize t_r

            t_r_list.append(t_r)
            q_r_list.append(q_r)
            P_r_list_of_lists.append(P_r_current)

            G_r_LS = mode_dot(_X_proj_for_svd, t_r.T, mode=0)
            W_G_r = _construct_milr_weight_tensor(G_r_LS.shape, self.Ln, self.alpha, self.N_modal_X, device)
            G_r = G_r_LS / (1.0 + self.lambda_X * W_G_r).clamp(min=1e-9) # Avoid division by near zero from weights
            G_r_list.append(G_r)

            u_r_vec = Fr_mat @ q_r
            d_r_LS = (t_r.T @ u_r_vec).squeeze()
            d_r = d_r_LS / (1.0 + self.lambda_Y)
            d_r_list.append(d_r.reshape(1))

            # --- W_r Calculation (Corrected logic for shapes) ---
            G_r_mat = matricize_n(G_r, mode=0) # Shape: 1 x K_core
            if G_r_mat.ndim == 1: G_r_mat = G_r_mat.unsqueeze(0) # Ensure 1 x K_core

            if P_r_current:
                P_kron = kronecker(P_r_current[::-1]) # Shape: (Prod I_k_non_sample) x K_core
                
                norm_G_r_mat_sq = torch.norm(G_r_mat)**2
                if norm_G_r_mat_sq > 1e-12:
                    G_r_vec_pinv = G_r_mat.T / norm_G_r_mat_sq # Shape: K_core x 1
                else:
                    G_r_vec_pinv = torch.zeros_like(G_r_mat.T)
                W_r_val = P_kron @ G_r_vec_pinv # Shape: (Prod I_k_non_sample) x 1
            elif G_r_mat.numel() == 1: # X is vector, G_r is scalar (1x1 matrix)
                g_val = G_r_mat.item()
                W_r_val = torch.tensor([[1.0/g_val if abs(g_val) > 1e-12 else 0.0]], device=device, dtype=X.dtype)
            else: # Should not be hit if model is fitting
                prod_X_non_sample_dims = X.shape[1:].numel() if X.ndim > 1 else 1
                W_r_val = torch.zeros((prod_X_non_sample_dims, 1), device=device, dtype=X.dtype) # Default to zero vector
            W_r_list.append(W_r_val)
            # --- End W_r Calculation ---

            Er = Er - tucker_to_tensor((G_r, [t_r] + P_r_current))
            Fr_mat = Fr_mat - d_r * (t_r @ q_r.T)

        self.P_r_all_components = P_r_list_of_lists
        self.Q_r_all_components = torch.cat(q_r_list, dim=1) if q_r_list else torch.empty((M_responses, 0), device=device)
        self.G_r_all_components = G_r_list
        self.D_r_all_components = torch.cat(d_r_list) if d_r_list else torch.empty(0, device=device)
        self.T_mat = torch.cat(t_r_list, dim=1) if t_r_list else torch.empty((I1_samples, 0), device=device)
        self.W_mat = torch.cat(W_r_list, dim=1) if W_r_list else torch.empty((X.shape[1:].numel() if X.ndim > 1 else 1, 0), device=device)
        
        return self

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> "HOPLS_MILR":
        X = X.to(device)
        Y = Y.to(device)

        if Y.ndim == 2:
            return self._fit_tensor_X_matrix_Y(X, Y)
        
        self._is_matrix_Y = False
        I1_samples = X.shape[0]
        self.N_modal_X = X.ndim - 1
        self.M_modal_Y = Y.ndim - 1
        self.actual_Km = self.Km_param if self.Km_param is not None else ([1] * self.M_modal_Y if self.M_modal_Y > 0 else [])


        if len(self.Ln) != self.N_modal_X and self.N_modal_X > 0:
            raise ValueError(f"Length of Ln ({len(self.Ln)}) must match X.ndim-1 ({self.N_modal_X}) for tensor X")
        if self.Km_param is not None and len(self.Km_param) != self.M_modal_Y and self.M_modal_Y > 0: # Check Km_param
            raise ValueError(f"Length of Km ({len(self.Km_param)}) must match Y.ndim-1 ({self.M_modal_Y}) for tensor Y")

        Er, Fr = X.clone(), Y.clone()

        P_r_list_of_lists: List[List[torch.Tensor]] = []
        Q_r_list_of_lists: List[List[torch.Tensor]] = []
        G_r_list: List[torch.Tensor] = []
        D_r_list: List[torch.Tensor] = []
        t_r_list: List[torch.Tensor] = []
        W_r_list: List[torch.Tensor] = []

        for r_component in range(self.R):
            if torch.norm(Er) < self.epsilon or torch.norm(Fr) < self.epsilon: # Use < for epsilon
                break

            Cr = torch.tensordot(Er, Fr, dims=([0], [0]))
            ranks_for_Cr_HOOI = self.Ln + self.actual_Km
            
            if not ranks_for_Cr_HOOI : # Both X and Y are vectors (ndim=1 for non-sample modes)
                 P_r_current = []
                 Q_r_current = []
            else:
                _, factors_Cr = tucker(Cr, rank=ranks_for_Cr_HOOI, init="svd", n_iter_max=15)
                P_r_current = factors_Cr[:self.N_modal_X]
                Q_r_current = factors_Cr[self.N_modal_X:]
            
            P_r_current = [p / torch.norm(p, dim=0, keepdim=True).clamp(min=1e-9) for p in P_r_current]
            Q_r_current = [q / torch.norm(q, dim=0, keepdim=True).clamp(min=1e-9) for q in Q_r_current]

            _X_proj_for_svd = Er
            if self.N_modal_X > 0:
                _X_proj_for_svd = multi_mode_dot(Er, [p.T for p in P_r_current], modes=list(range(1, self.N_modal_X + 1)))
            
            U_tr, _, _ = torch.linalg.svd(matricize_n(_X_proj_for_svd, mode=0), full_matrices=False)
            t_r = U_tr[:, 0:1]
            if torch.norm(t_r) > 1e-9: t_r = t_r / torch.norm(t_r)

            t_r_list.append(t_r)
            P_r_list_of_lists.append(P_r_current)
            Q_r_list_of_lists.append(Q_r_current)

            G_r_LS = mode_dot(_X_proj_for_svd, t_r.T, mode=0)
            W_G_r = _construct_milr_weight_tensor(G_r_LS.shape, self.Ln, self.alpha, self.N_modal_X, device)
            G_r = G_r_LS / (1.0 + self.lambda_X * W_G_r).clamp(min=1e-9)
            G_r_list.append(G_r)

            _Y_proj_for_Dr = Fr
            if self.M_modal_Y > 0:
                _Y_proj_for_Dr = multi_mode_dot(Fr, [q.T for q in Q_r_current], modes=list(range(1, self.M_modal_Y + 1)))
            D_r_LS = mode_dot(_Y_proj_for_Dr, t_r.T, mode=0)
            W_D_r = _construct_milr_weight_tensor(D_r_LS.shape, self.actual_Km, self.alpha, self.M_modal_Y, device)
            D_r = D_r_LS / (1.0 + self.lambda_Y * W_D_r).clamp(min=1e-9)
            D_r_list.append(D_r)
            
            # --- W_r Calculation (Corrected logic for shapes) ---
            G_r_mat = matricize_n(G_r, mode=0) # Shape: 1 x K_core
            if G_r_mat.ndim == 1: G_r_mat = G_r_mat.unsqueeze(0) # Ensure 1 x K_core

            if P_r_current:
                P_kron = kronecker(P_r_current[::-1]) # Shape: (Prod I_k_non_sample) x K_core
                
                norm_G_r_mat_sq = torch.norm(G_r_mat)**2
                if norm_G_r_mat_sq > 1e-12:
                    G_r_vec_pinv = G_r_mat.T / norm_G_r_mat_sq # Shape: K_core x 1
                else:
                    G_r_vec_pinv = torch.zeros_like(G_r_mat.T)
                W_r_val = P_kron @ G_r_vec_pinv # Shape: (Prod I_k_non_sample) x 1
            elif G_r_mat.numel() == 1: # X is vector, G_r is scalar (1x1 matrix)
                g_val = G_r_mat.item()
                W_r_val = torch.tensor([[1.0/g_val if abs(g_val) > 1e-12 else 0.0]], device=device, dtype=X.dtype)
            else:
                prod_X_non_sample_dims = X.shape[1:].numel() if X.ndim > 1 else 1
                W_r_val = torch.zeros((prod_X_non_sample_dims, 1), device=device, dtype=X.dtype)
            W_r_list.append(W_r_val)
            # --- End W_r Calculation ---

            Er = Er - tucker_to_tensor((G_r, [t_r] + P_r_current))
            Fr = Fr - tucker_to_tensor((D_r, [t_r] + Q_r_current))
        
        self.P_r_all_components = P_r_list_of_lists
        self.Q_r_all_components = Q_r_list_of_lists
        self.G_r_all_components = G_r_list
        self.D_r_all_components = D_r_list
        self.T_mat = torch.cat(t_r_list, dim=1) if t_r_list else torch.empty((I1_samples,0), device=device)
        self.W_mat = torch.cat(W_r_list, dim=1) if W_r_list else torch.empty((X.shape[1:].numel() if X.ndim > 1 else 1, 0), device=device)
        
        return self
    
    def predict(
        self, X_new: torch.Tensor, Y_true_for_shape: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, int, List[float]]:
        if self.T_mat is None or self.W_mat is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        num_components_actually_fitted = self.W_mat.shape[1] # R can be larger than what was fitted
        if num_components_actually_fitted == 0:
            # Determine a sensible zero prediction shape
            pred_shape = list(X_new.shape[:1]) 
            if Y_true_for_shape is not None:
                pred_shape.extend(Y_true_for_shape.shape[1:])
            elif self._is_matrix_Y:
                # self.Q_r_all_components is M x R_fitted for matrix Y
                q_data = self.Q_r_all_components # This is Q_mat: M x R_fitted
                pred_shape.append(q_data.shape[0] if q_data.ndim == 2 and q_data.shape[0] > 0 else 1)
            else: # Tensor Y
                if self.D_r_all_components: # List of D_r tensors
                    pred_shape.extend(self.D_r_all_components[0].shape[1:])
                else: 
                    pred_shape.append(1)
            return torch.zeros(pred_shape, device=device, dtype=X_new.dtype), 0, []

        X_new_dev = X_new.to(device)
        X_new_mat = matricize_n(X_new_dev, mode=0) # Shape: (I1_new, Prod_X_non_sample_dims)
        
        # T_new_all_r = X_new_mat @ W_mat. W_mat is (Prod_X_non_sample_dims x R_fitted)
        T_new_all_r_mat = X_new_mat @ self.W_mat # Shape: (I1_new, R_fitted)

        best_q2_val = -float('inf')
        best_Y_pred_val: Optional[torch.Tensor] = None
        best_r_val = 0
        q2s_list: List[float] = []

        Y_true_dev: Optional[torch.Tensor] = None
        output_Y_shape = list(X_new_dev.shape[:1]) # Start with (n_samples_new, ...)
        if Y_true_for_shape is not None:
            Y_true_dev = Y_true_for_shape.to(device)
            output_Y_shape.extend(Y_true_for_shape.shape[1:])
        elif self._is_matrix_Y:
            output_Y_shape.append(self.Q_r_all_components.shape[0] if self.Q_r_all_components.ndim == 2 and self.Q_r_all_components.shape[0] > 0 else 1)
        else: # Tensor Y
            if self.D_r_all_components: output_Y_shape.extend(self.D_r_all_components[0].shape[1:])
            else: output_Y_shape.append(1)


        for r_to_use in range(1, num_components_actually_fitted + 1):
            T_new_current_r = T_new_all_r_mat[:, :r_to_use] # I1_new x r_to_use

            if self._is_matrix_Y:
                # self.Q_r_all_components is Q_mat (M x R_fitted)
                # self.D_r_all_components is d_coeffs (R_fitted vector)
                Q_r_subset = self.Q_r_all_components[:, :r_to_use] # M x r_to_use
                D_r_diag_subset = torch.diag(self.D_r_all_components[:r_to_use]) # r_to_use x r_to_use
                
                # Y_pred = T_new @ D @ Q^T
                Y_pred_current_iter = T_new_current_r @ D_r_diag_subset @ Q_r_subset.T # (I1_new x M)
            else: # Tensor Y
                # self.Q_r_all_components is List (over r) of Lists (Q_r^(j))
                # self.D_r_all_components is List (over r) of D_r tensors (1, K2, ..., KM)
                
                Y_pred_current_iter = torch.zeros(output_Y_shape, device=device, dtype=X_new.dtype)
                for i_comp_idx in range(r_to_use):
                    t_new_for_comp = T_new_current_r[:, i_comp_idx:i_comp_idx+1] # I1_new x 1
                    D_comp = self.D_r_all_components[i_comp_idx]                 # (1, K2, ..., KM)
                    Q_comp_list = self.Q_r_all_components[i_comp_idx]            # List [Q_comp^(1), ..., Q_comp^(M-1)]
                    
                    term_Y_pred = tucker_to_tensor((D_comp, [t_new_for_comp] + Q_comp_list))
                    Y_pred_current_iter = Y_pred_current_iter + term_Y_pred
            
            if Y_true_dev is not None and Y_pred_current_iter.numel() > 0 :
                # Reshape for metric if Y_true_dev and Y_pred_current_iter are not flat
                q2_val = self.metric(
                    Y_true_dev.contiguous().view(Y_true_dev.shape[0], -1),
                    Y_pred_current_iter.contiguous().view(Y_pred_current_iter.shape[0], -1)
                )
                q2s_list.append(q2_val)
                if q2_val > best_q2_val:
                    best_q2_val = q2_val
                    best_Y_pred_val = Y_pred_current_iter
                    best_r_val = r_to_use
            elif r_to_use == num_components_actually_fitted and Y_true_dev is None: # No Y_true, use all fitted components
                best_Y_pred_val = Y_pred_current_iter
                best_r_val = r_to_use
        
        if best_Y_pred_val is None: # Fallback if loop didn't run or no best Q2 found
            if num_components_actually_fitted > 0 and 'Y_pred_current_iter' in locals() and Y_pred_current_iter.numel() > 0:
                best_Y_pred_val = Y_pred_current_iter 
                best_r_val = num_components_actually_fitted
            else: # Truly no components or prediction possible
                 best_Y_pred_val = torch.zeros(output_Y_shape, device=device, dtype=X_new.dtype)
                 best_r_val = 0

        return best_Y_pred_val, best_r_val, q2s_list

    def score(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        if self.T_mat is None: 
             self.fit(X,Y)

        _, _, q2s = self.predict(X, Y_true_for_shape=Y)
        if not q2s:
            # If model could not make a prediction or q2s is empty, return a very low score
            # Check if Y is all zeros or constant, which might lead to sst=0 and problematic Q2
            if torch.allclose(Y, torch.zeros_like(Y)) or torch.allclose(Y, Y[0].expand_as(Y)):
                 # If Y is all zeros/constant, and prediction is also zero/constant, Q2 can be 1 or 0.
                 # If prediction is different, Q2 can be negative.
                 # This case often indicates no variance to explain.
                 # For simplicity, if no q2s, assume worst case.
                 return -float('inf') 
            return -float('inf') # Default for no valid Q2 scores
        return max(q2s) if q2s else -float('inf') # Ensure max() is not called on empty list    



    def score(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        # The predict method now finds the best_r based on Q2 score using Y
        Y_pred, best_r, q2s = self.predict(X, Y_true_for_shape=Y)
        if not q2s: # No components or Q2 could not be calculated
            return -float('inf') # Or some other indicator of failure / no prediction
        return max(q2s) # Return the best Q2 achieved
    
    
    # def predict(
    #     self, X_new: torch.Tensor, Y_true_for_shape: Optional[torch.Tensor] = None
    # ) -> Tuple[torch.Tensor, int, List[float]]:
    #     if self.T_mat is None or self.W_mat is None : # Check if model is fitted
    #         raise ValueError("Model not fitted yet. Call fit() first.")
    #     if self.T_mat.numel() == 0 : # No components were fitted
    #         if Y_true_for_shape is not None:
    #             return torch.zeros_like(Y_true_for_shape.to(device)), 0, []
    #         else: # Cannot determine output shape, return empty or raise error
    #             return torch.empty(0, device=device), 0, []


    #     X_new_dev = X_new.to(device)
    #     X_new_mat = matricize_n(X_new_dev, mode=0) # I1_new x (Prod X_modes_non_sample)
        
    #     # T_new = X_new_mat @ W_mat but W_mat could be empty if R=0
    #     if self.W_mat.numel() == 0 or self.W_mat.shape[1] == 0 : # W_mat could be (K,0)
    #          T_new_mat = torch.empty((X_new_dev.shape[0], 0), device=device)
    #     else:
    #          T_new_mat = X_new_mat @ self.W_mat # I1_new x R_fitted

    #     num_fitted_components = T_new_mat.shape[1]
        
    #     # Iterate r from 1 to num_fitted_components to find best Q2, like original code
    #     best_q2_val = -float('inf')
    #     best_Y_pred_val: Optional[torch.Tensor] = None
    #     best_r_val = 0
    #     q2s_list: List[float] = []

    #     # Determine shape of Y_true for Q2 calculation, if provided
    #     Y_true_dev: Optional[torch.Tensor] = None
    #     if Y_true_for_shape is not None:
    #         Y_true_dev = Y_true_for_shape.to(device)

    #     for r_to_predict in range(1, num_fitted_components + 1):
    #         current_T_new = T_new_mat[:, :r_to_predict]

    #         if self._is_matrix_Y:
    #             Q_mat_r = self.Q_r_all_components[:, :r_to_predict] # M x r_to_predict
    #             D_diag_r = torch.diag(self.D_r_all_components[:r_to_predict]) # r_to_predict x r_to_predict
    #             Y_pred_current_r = current_T_new @ D_diag_r @ Q_mat_r.T
    #         else: # Tensor Y
    #             # Y_pred = sum_{i=1 to r_to_predict} [[ D_i; t_new_i, Q_i^(1), ..., Q_i^(M-1) ]]
    #             # D_i is self.D_r_all_components[i-1]
    #             # t_new_i is current_T_new[:, i-1:i]
    #             # Q_i are self.Q_r_all_components[i-1]
                
    #             # Determine Y_pred shape from the first D_r and Y_true_for_shape or X_new
    #             if self.D_r_all_components:
    #                 Y_pred_shape_first_mode = X_new_dev.shape[0]
    #                 Y_pred_shape_other_modes = self.D_r_all_components[0].shape[1:]
    #                 Y_pred_current_r = torch.zeros((Y_pred_shape_first_mode, *Y_pred_shape_other_modes), device=device)
                    
    #                 for i_comp in range(r_to_predict):
    #                     t_new_i_comp = current_T_new[:, i_comp:i_comp+1]
    #                     D_i_comp = self.D_r_all_components[i_comp]
    #                     Q_i_comp_list = self.Q_r_all_components[i_comp]
                        
    #                     term_Y_pred = tucker_to_tensor((D_i_comp, [t_new_i_comp] + Q_i_comp_list))
    #                     Y_pred_current_r = Y_pred_current_r + term_Y_pred
    #             else: # No D components, should not happen if num_fitted_components > 0
    #                 Y_pred_current_r = torch.empty(0, device=device)


    #         if Y_true_dev is not None and Y_pred_current_r.numel() > 0:
    #             q2_val = self.metric(
    #                 Y_true_dev.reshape(Y_true_dev.shape[0], -1),
    #                 Y_pred_current_r.reshape(Y_pred_current_r.shape[0], -1)
    #             )
    #             q2s_list.append(q2_val)
    #             if q2_val > best_q2_val:
    #                 best_q2_val = q2_val
    #                 best_Y_pred_val = Y_pred_current_r
    #                 best_r_val = r_to_predict
    #         elif r_to_predict == num_fitted_components : # If no Y_true, just use all components
    #             best_Y_pred_val = Y_pred_current_r
    #             best_r_val = r_to_predict


    #     if best_Y_pred_val is None and num_fitted_components > 0: # Fallback if Q2 was always -inf but loop ran
    #         best_Y_pred_val = Y_pred_current_r # Use the last computed prediction
    #         best_r_val = num_fitted_components
    #     elif best_Y_pred_val is None and num_fitted_components == 0: # No components, predict zeros
    #          if Y_true_for_shape is not None:
    #             best_Y_pred_val = torch.zeros_like(Y_true_for_shape.to(device))
    #          elif self._is_matrix_Y: # Use Y shape from training if available (via Q_r_all_components)
    #             if isinstance(self.Q_r_all_components, torch.Tensor) and self.Q_r_all_components.numel()>0:
    #                 best_Y_pred_val = torch.zeros((X_new.shape[0], self.Q_r_all_components.shape[0]), device=device)
    #             else: best_Y_pred_val = torch.zeros((X_new.shape[0], 1), device=device) # Default
    #          else: # Tensor Y
    #             if self.D_r_all_components:
    #                 Y_pred_shape_other_modes = self.D_r_all_components[0].shape[1:]
    #                 best_Y_pred_val = torch.zeros((X_new.shape[0], *Y_pred_shape_other_modes), device=device)
    #             else: best_Y_pred_val = torch.zeros((X_new.shape[0], 1), device=device) # Default


    #     return best_Y_pred_val, best_r_val, q2s_list
    # def _fit_tensor_X_matrix_Y(self, X: torch.Tensor, Y: torch.Tensor) -> "HOPLS_MILR":
    #     self._is_matrix_Y = True
    #     I1_samples = X.shape[0]
    #     M_responses = Y.shape[1]
    #     self.N_modal_X = X.ndim - 1 # Number of P_r^(j) loading matrices for X
        
    #     if len(self.Ln) != self.N_modal_X and self.N_modal_X > 0:
    #          raise ValueError(f"Length of Ln ({len(self.Ln)}) must match X.ndim-1 ({self.N_modal_X})")

    #     Er, Fr_mat = X.clone(), Y.clone()

    #     # Initialize lists for storing component-wise results
    #     P_r_list_of_lists: List[List[torch.Tensor]] = []
    #     q_r_list: List[torch.Tensor] = [] # For Q_loadings_mat
    #     G_r_list: List[torch.Tensor] = []
    #     d_r_list: List[torch.Tensor] = [] # For D_coeffs_vec
    #     t_r_list: List[torch.Tensor] = []
    #     W_r_list: List[torch.Tensor] = []


    #     for r_component in range(self.R):
    #         if torch.norm(Er) <= self.epsilon or torch.norm(Fr_mat) <= self.epsilon:
    #             break

    #         Cr = mode_dot(Er, Fr_mat.T, mode=0)
            
    #         ranks_for_Cr_HOOI = [1] + self.Ln # Rank 1 for Y-related mode, Ln for X-related modes
    #         Gr_C, factors_Cr = tucker(Cr, rank=ranks_for_Cr_HOOI, init="svd", n_iter_max=15)
            
    #         q_r = factors_Cr[0] / torch.norm(factors_Cr[0]) # M x 1
    #         P_r_current = [p / torch.norm(p, dim=0, keepdim=True).clamp(min=1e-9) for p in factors_Cr[1:]]

    #         _X_proj_for_svd = Er
    #         if self.N_modal_X > 0:
    #             _X_proj_for_svd = multi_mode_dot(Er, [p.T for p in P_r_current], modes=list(range(1, self.N_modal_X + 1)))
            
    #         U_tr, _, _ = torch.linalg.svd(matricize_n(_X_proj_for_svd, mode=0), full_matrices=False)
    #         t_r = U_tr[:, 0:1] # I1 x 1
    #         t_r = t_r / torch.norm(t_r)

    #         t_r_list.append(t_r)
    #         q_r_list.append(q_r)
    #         P_r_list_of_lists.append(P_r_current)

    #         G_r_LS = mode_dot(_X_proj_for_svd, t_r.T, mode=0) # Shape (1, L2, ..., LN)
    #         W_G_r = _construct_milr_weight_tensor(G_r_LS.shape, self.Ln, self.alpha, self.N_modal_X, device)
    #         G_r = G_r_LS / (1.0 + self.lambda_X * W_G_r)
    #         G_r_list.append(G_r)

    #         u_r_vec = Fr_mat @ q_r # I1 x 1
    #         d_r_LS = (t_r.T @ u_r_vec).squeeze()
    #         d_r = d_r_LS / (1.0 + self.lambda_Y)
    #         d_r_list.append(d_r.reshape(1))

    #         # Calculate W_r for prediction: W_r = kron(P_r reversed) @ pinv(vec(G_r))
    #         # matricize(G_r) is 1 x (prod Lk). We need its pseudoinverse as (prod Lk) x 1.
    #         # The Kronecker product part needs P_r components.
    #         if P_r_current: # If there are P_r matrices
    #             P_kron = kronecker(P_r_current[::-1]) # Reversed order for kronecker as in original code
    #             G_r_vec_pinv = torch.linalg.pinv(matricize_n(G_r, mode=0).T) # (prod Lk) x 1
    #             W_r = P_kron @ G_r_vec_pinv # (Prod X_modes_non_sample) x 1
    #         else: # X is a matrix, G_r is 1xL2 (L2=I2). P_r_current is effectively P^(1)
    #               # This case needs refinement, W_r should be I2x1 if G_r is 1xI2
    #               # W_r = P_r^(1) @ pinv(G_r.T). No, that's not right.
    #               # If X is I1xI2, Er is I1xI2, G_r_LS = t_r.T @ Er (1xI2)
    #               # t_r = Er @ W_r. So W_r = pinv(Er) @ t_r ? No.
    #               # W_r should be related to P_r and G_r such that X_unfold @ W_r = t_r
    #               # From original: W_r = kronecker(P_r[::-1]) @ ridge_pinv(matricize(G_d), self.lam)
    #               # Let's assume the structure of W_r from your original code for consistency,
    #               # where G_d was G_r_LS. Now G_r is regularized.
    #             if P_r_current:
    #                 P_kron = kronecker(P_r_current[::-1])
    #                 # For pinv of a row vector (matricize(G_r)), it's G_r.T / ||G_r||^2
    #                 G_r_mat = matricize_n(G_r, mode=0) # 1 x K
    #                 if torch.norm(G_r_mat) > 1e-9:
    #                     G_r_vec_pinv = G_r_mat.T / (torch.norm(G_r_mat)**2)
    #                 else:
    #                     G_r_vec_pinv = torch.zeros_like(G_r_mat.T)
    #                 W_r = P_kron @ G_r_vec_pinv
    #             elif G_r.numel() > 0 : # X is vector, G_r is scalar
    #                 W_r = torch.tensor([[1.0/G_r.item() if G_r.item() !=0 else 0.0]], device=device) # Effectively 1/G_r
    #             else: # Should not happen if components are found
    #                 W_r = torch.empty((X.shape[1:].numel(),0), device=device)

    #         W_r_list.append(W_r)

    #         Er = Er - tucker_to_tensor((G_r, [t_r] + P_r_current))
    #         Fr_mat = Fr_mat - d_r * (t_r @ q_r.T)

    #     self.P_r_all_components = P_r_list_of_lists
    #     self.Q_r_all_components = torch.cat(q_r_list, dim=1) if q_r_list else torch.empty((M_responses, 0), device=device)
    #     self.G_r_all_components = G_r_list
    #     self.D_r_all_components = torch.cat(d_r_list) if d_r_list else torch.empty(0, device=device)
    #     self.T_mat = torch.cat(t_r_list, dim=1) if t_r_list else torch.empty((I1_samples, 0), device=device)
    #     self.W_mat = torch.cat(W_r_list, dim=1) if W_r_list else torch.empty((X.shape[1:].numel(),0), device=device)
        
    #     return self

    # def fit(self, X: torch.Tensor, Y: torch.Tensor) -> "HOPLS_MILR":
    #     X = X.to(device)
    #     Y = Y.to(device)

    #     if Y.ndim == 2:
    #         return self._fit_tensor_X_matrix_Y(X, Y)
        
    #     self._is_matrix_Y = False
    #     I1_samples = X.shape[0]
    #     self.N_modal_X = X.ndim - 1
    #     self.M_modal_Y = Y.ndim - 1
    #     self.actual_Km = self.Km_param if self.Km_param is not None else [1] * self.M_modal_Y # Fallback for Km

    #     if len(self.Ln) != self.N_modal_X and self.N_modal_X > 0:
    #         raise ValueError(f"Length of Ln ({len(self.Ln)}) must match X.ndim-1 ({self.N_modal_X}) for tensor X")
    #     if self.Km_param is not None and len(self.Km_param) != self.M_modal_Y and self.M_modal_Y > 0:
    #         raise ValueError(f"Length of Km ({len(self.Km_param)}) must match Y.ndim-1 ({self.M_modal_Y}) for tensor Y")


    #     Er, Fr = X.clone(), Y.clone()

    #     P_r_list_of_lists: List[List[torch.Tensor]] = []
    #     Q_r_list_of_lists: List[List[torch.Tensor]] = []
    #     G_r_list: List[torch.Tensor] = []
    #     D_r_list: List[torch.Tensor] = []
    #     t_r_list: List[torch.Tensor] = []
    #     W_r_list: List[torch.Tensor] = [] # For X-weights

    #     for r_component in range(self.R):
    #         if torch.norm(Er) <= self.epsilon or torch.norm(Fr) <= self.epsilon:
    #             break

    #         Cr = torch.tensordot(Er, Fr, dims=([0], [0]))
            
    #         ranks_for_Cr_HOOI = self.Ln + self.actual_Km # Ranks for modes I2..IN, then J2..JM
    #         if not ranks_for_Cr_HOOI: #Both X and Y are vectors
    #              P_r_current = []
    #              Q_r_current = []
    #         else:
    #             _, factors_Cr = tucker(Cr, rank=ranks_for_Cr_HOOI, init="svd", n_iter_max=15)
    #             P_r_current = factors_Cr[:self.N_modal_X]
    #             Q_r_current = factors_Cr[self.N_modal_X:]

    #         P_r_current = [p / torch.norm(p, dim=0, keepdim=True).clamp(min=1e-9) for p in P_r_current]
    #         Q_r_current = [q / torch.norm(q, dim=0, keepdim=True).clamp(min=1e-9) for q in Q_r_current]

    #         _X_proj_for_svd = Er
    #         if self.N_modal_X > 0:
    #             _X_proj_for_svd = multi_mode_dot(Er, [p.T for p in P_r_current], modes=list(range(1, self.N_modal_X + 1)))
            
    #         U_tr, _, _ = torch.linalg.svd(matricize_n(_X_proj_for_svd, mode=0), full_matrices=False)
    #         t_r = U_tr[:, 0:1]
    #         t_r = t_r / torch.norm(t_r)

    #         t_r_list.append(t_r)
    #         P_r_list_of_lists.append(P_r_current)
    #         Q_r_list_of_lists.append(Q_r_current)

    #         G_r_LS = mode_dot(_X_proj_for_svd, t_r.T, mode=0)
    #         W_G_r = _construct_milr_weight_tensor(G_r_LS.shape, self.Ln, self.alpha, self.N_modal_X, device)
    #         G_r = G_r_LS / (1.0 + self.lambda_X * W_G_r)
    #         G_r_list.append(G_r)

    #         _Y_proj_for_Dr = Fr
    #         if self.M_modal_Y > 0:
    #             _Y_proj_for_Dr = multi_mode_dot(Fr, [q.T for q in Q_r_current], modes=list(range(1, self.M_modal_Y + 1)))
    #         D_r_LS = mode_dot(_Y_proj_for_Dr, t_r.T, mode=0)
    #         W_D_r = _construct_milr_weight_tensor(D_r_LS.shape, self.actual_Km, self.alpha, self.M_modal_Y, device)
    #         D_r = D_r_LS / (1.0 + self.lambda_Y * W_D_r)
    #         D_r_list.append(D_r)
            
    #         # Calculate W_r (X-weights for prediction)
    #         if P_r_current:
    #             P_kron = kronecker(P_r_current[::-1])
    #             G_r_mat = matricize_n(G_r, mode=0) # 1 x K
    #             if torch.norm(G_r_mat) > 1e-9:
    #                 G_r_vec_pinv = G_r_mat.T / (torch.norm(G_r_mat)**2) # pinv for row vector
    #             else:
    #                 G_r_vec_pinv = torch.zeros_like(G_r_mat.T)
    #             W_r_val = P_kron @ G_r_vec_pinv
    #         elif G_r.numel() > 0: # X is vector, G_r scalar
    #             W_r_val = torch.tensor([[1.0/G_r.item() if G_r.item()!=0 else 0.0]], device=device)
    #         else:
    #             W_r_val = torch.empty((X.shape[1:].numel(),0), device=device)
    #         W_r_list.append(W_r_val)


    #         Er = Er - tucker_to_tensor((G_r, [t_r] + P_r_current))
    #         Fr = Fr - tucker_to_tensor((D_r, [t_r] + Q_r_current))
        
    #     self.P_r_all_components = P_r_list_of_lists
    #     self.Q_r_all_components = Q_r_list_of_lists
    #     self.G_r_all_components = G_r_list
    #     self.D_r_all_components = D_r_list
    #     self.T_mat = torch.cat(t_r_list, dim=1) if t_r_list else torch.empty((I1_samples,0), device=device)
    #     self.W_mat = torch.cat(W_r_list, dim=1) if W_r_list else torch.empty((X.shape[1:].numel(),0), device=device)
        
    #     return self