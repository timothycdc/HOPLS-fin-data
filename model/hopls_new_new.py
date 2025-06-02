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

class HOPLS:
    def __init__(
        self,
        R: int,
        Ln: List[int],
        Km: Optional[List[int]] = None,
        metric: Optional[callable] = None,
        epsilon: float = 1e-6,
    ) -> None:
        """
        Higher-Order Partial Least Squares (HOPLS).

        Parameters:
            R       : int             – number of latent components.
            Ln      : list of int     – Tucker ranks for modes 2…N of X.
            Km      : list of int     – Tucker ranks for modes 2…M of Y.
            metric  : callable        – scoring metric (default: qsquared).
            epsilon : float           – deflation stopping threshold.
        """
        self.R = R
        self.Ln = Ln
        self.Km = Km 
        self.metric = metric or qsquared
        self.epsilon = epsilon

        # number of non-sample modes
        self.N = len(self.Ln)
        self.M = len(self.Km) if Km is not None else 2

        # will hold (P, Q, D_list, T, W) after fit
        self.model: Optional[Tuple] = None

    def _fit_2d(self, X: torch.Tensor, Y: torch.Tensor) -> "HOPLS":
        """
        Special case when Y is a matrix (mode-2 tensor): Algorithm 2 in Zhao et al. (2012).
        """
        # NEW: ensure data on correct device
        Er, Fr = X.clone().to(device), Y.clone().to(device)
        P: List[List[torch.Tensor]] = []
        Q: List[torch.Tensor] = []
        D = torch.zeros((self.R, self.R), dtype=X.dtype, device=device)
        T: List[torch.Tensor] = []
        W: List[torch.Tensor] = []

        for r in range(self.R):
            # stopping criterion (||Er||, ||Fr|| > ε)
            if torch.norm(Er) <= self.epsilon or torch.norm(Fr) <= self.epsilon:
                break

            # 2. form covariance tensor C_r = mode_dot(Er, Frᵀ, mode=0)
            Cr = mode_dot(Er, Fr.t(), mode=0)

            # 3. perform HOOI‐based Tucker of C_r with ranks [1] + Ln
            #    → core GrC and loading factors [q_r, P_r¹,…,P_rᴺ]
            GrC, latent_factors = tucker(Cr, rank=[1] + self.Ln)
            # normalise
            q_r = latent_factors[0] / torch.norm(latent_factors[0])
            P_r = [p / torch.norm(p) for p in latent_factors[1:]]

            # 4. compute latent vector t_r by projecting Er onto P_r loadings:
            #    t̃ = Er ×ₙ P_r⁽ᵀ⁾, then t_r = matricize(t̃)·pinv(mat(GrC)), normalised
            t_tilde = multi_mode_dot(
                Er, P_r, modes=list(range(1, len(P_r) + 1)), transpose=True
            )
            GrC_pinv = torch.linalg.pinv(matricize(GrC))
            t_r = torch.mm(matricize(t_tilde), GrC_pinv)
            t_r = t_r / torch.norm(t_r)

            # 5. scalar regression from t_r to Y:  u_r = Fr·q_r,  d_r = u_rᵀ·t_r
            u_r = torch.mm(Fr, q_r)
            d_r = torch.mm(u_r.t(), t_r)
            D[r, r] = d_r

            # 6. build W_r = kron(P_r^(N)…P_r^(1)) · pinv(mat( G_d ) )
            #    where G_d = Er ×₁ t_rᵀ ×₂ P_r^(1)ᵀ … ×ₙ P_r^(N)ᵀ
            G_d = tucker_to_tensor((Er, [t_r] + P_r), transpose_factors=True)
            G_d_pinv = torch.linalg.pinv(matricize(G_d))
            W_r = torch.mm(kronecker(P_r[::-1]), G_d_pinv)

            # 7. deflate: Er ← Er – G_d ×₁ t_r ×₂ P_r¹ … ×ₙ P_rᴺ
            #            Fr ← Fr – d_r·t_r·q_rᵀ
            Er = Er - tucker_to_tensor((G_d, [t_r] + P_r))
            Fr = Fr - d_r * torch.mm(t_r, q_r.t())

            # collect
            P.append(P_r)
            Q.append(q_r)
            T.append(t_r)
            W.append(W_r)

        # stack into final matrices
        Q_mat = torch.cat(Q, dim=1)
        T_mat = torch.cat(T, dim=1)
        W_mat = torch.cat(W, dim=1)

        self.model = (P, Q_mat, D, T_mat, W_mat)
        return self

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> "HOPLS":
        """
        Full HOPLS fit for X.ndim≥3, Y.ndim≥2 (Algorithm 1).
        """
        assert X.ndim >= 3, "X must be at least 3-way."
        assert Y.ndim >= 2, "Y must be at least matrix (2-way)."
        assert len(self.Ln) == X.ndim - 1

        # if Y is 2-way, use the special routine
        if Y.ndim == 2:
            return self._fit_2d(X, Y)

        assert (
            len(self.Km) == Y.ndim - 1
        ), f"The list of ranks for the decomposition of Y (Km) need to be equal to the mode of Y -1: {Y.ndim-1}."

        # NEW: ensure data on chosen device
        Er, Fr = X.clone().to(device), Y.clone().to(device)
        P: List[List[torch.Tensor]] = []
        Q: List[List[torch.Tensor]] = []
        G_list: List[torch.Tensor] = []
        D_list: List[torch.Tensor] = []
        T: List[torch.Tensor] = []
        W: List[torch.Tensor] = []

        for r in range(self.R):
            if torch.norm(Er) <= self.epsilon or torch.norm(Fr) <= self.epsilon:
                break

            # 1. form full cross-covariance C_r = ∑ₖ Er[...,k]·Fr[...,k]
            # NEW: pure-Torch tensordot instead of NumPy
            Cr = torch.tensordot(Er, Fr, dims=([0], [0]))

            # 2. Tucker‐HOOI on C_r with ranks Ln+Km → factors for X‐side (Pr) and Y‐side (Qr)
            _, latent_factors = tucker(Cr, rank=self.Ln + self.Km)
            Pr = latent_factors[: self.N]
            Qr = latent_factors[self.N :]

            # 3. compute t_r = leading left SVD vector of Er ×ₙ P_r^(n)ᵀ
            Er_proj = multi_mode_dot(
                Er, Pr, modes=list(range(1, self.N + 1)), transpose=True
            )
            U, _, _ = torch.linalg.svd(matricize(Er_proj))  # NEW: torch.linalg.svd
            t_r = U[:, :1]

            # 4. core tensors:
            #    G_r = Er ×₁ t_rᵀ ×₂ Pr¹ᵀ ×₃ ... ×ₙ Pr^(N-1)ᵀ
            #    D_r = Fr ×₁ t_rᵀ ×₂ Qr¹ᵀ ×₃ ... ×ₘ Qr^(M-1)ᵀ
            G_r = tucker_to_tensor((Er, [t_r] + Pr), transpose_factors=True)
            D_r = tucker_to_tensor((Fr, [t_r] + Qr), transpose_factors=True)

            # 5. compute W_r = kron(Pr[::-1]) · pinv( mat(G_r) )
            G_r_pinv = torch.linalg.pinv(matricize(G_r))
            W_r = torch.mm(kronecker(Pr[::-1]), G_r_pinv)

            # 6. deflate both Er and Fr
            Er = Er - tucker_to_tensor((G_r, [t_r] + Pr))
            Fr = Fr - tucker_to_tensor((D_r, [t_r] + Qr))

            # 7.collect
            P.append(Pr)
            Q.append(Qr)
            G_list.append(G_r)
            D_list.append(D_r)
            T.append(t_r)
            W.append(W_r)

        # stack T and W
        T_mat = torch.cat(T, dim=1)
        W_mat = torch.cat(W, dim=1)
        self.model = (P, Q, D_list, T_mat, W_mat)
        return self

    def predict(
        self, X: torch.Tensor, Y: torch.Tensor
    ) -> Tuple[torch.Tensor, int, List[float]]:
        """Compute the HOPLS for X and Y with respect to the parameters R, Ln and Km.
        
            Returns:
            Y_pred : Tensor of shape matching Y (for 2d) or X-mode-0 shape + Y-modes
            best_r : int (always self.R here)
            q2s    : empty list (Q² not computed in ridge version)
        """
        import warnings

        # Unpack model parameters: Q holds loadings for Y, D holds diagonal matrices, W holds weights for X
        _, Q, D, _, W = self.model

        # initialize best trackers
        best_q2 = -np.inf
        best_Y_pred = None
        best_r = 0

        # Precompute Y-side projection blocks for higher-order Y
        if Y.ndim > 2:
            Q_star = []
            for r in range(self.R):
                # Build the Kronecker product of the r-th factor matrices of Q
                Qkron = kronecker([Q[r][self.M - m - 1] for m in range(self.M)])
                Q_star.append(
                    torch.mm(matricize(D[r][None, ...]), Qkron.t())  # D[r] as 2D
                )
            Q_star = torch.cat(Q_star)

        q2s = []
        for r in range(1, self.R + 1):
            # Special case: two-way Y
            if Y.ndim == 2:
                Q_star = torch.mm(D[:r, :r], Q[:, :r].t())

            # inner projection: X-weights times Y-loadings
            inter = torch.mm(W[:, :r], Q_star[:r])

            # matricise X (pure-torch) then project
            Z = torch.mm(matricize(X), inter)

            # use tensorly.fold to invert the unfolding in Fortran-order
            Y_pred = fold(Z, mode=0, shape=Y.shape)

            # Compute Q²
            Q2 = qsquared(Y, Y_pred)
            q2s.append(Q2)

            if Q2 > best_q2:
                best_q2 = Q2
                best_r = r
                best_Y_pred = Y_pred

        # fallback if no component improved
        if best_Y_pred is None:
            # warnings.warn(
            #     "HOPLS.predict: no optimal component found, using last Y_pred"
            # )
            best_Y_pred = Y_pred
            best_r = r

        return best_Y_pred, best_r, q2s

    def score(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """Fit + predict and return the chosen metric."""
        self.fit(X, Y)
        Y_pred, _ = self.predict(X, Y)
        return self.metric(
            Y.reshape(Y.shape[0], -1), Y_pred.reshape(Y_pred.shape[0], -1)
        )


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
    
    
import torch
import tensorly as tl
from tensorly import tucker_to_tensor, fold
from tensorly.decomposition import tucker
from tensorly.tenalg import kronecker, mode_dot, multi_mode_dot
from typing import List, Optional, Tuple, Union

# Use PyTorch backend
tl.set_backend("pytorch")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


def matricize_n(tensor: torch.Tensor, mode: int = 0) -> torch.Tensor:
    """Unfolds a tensor into a matrix along a specified mode."""
    return tl.unfold(tensor, mode)

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

    if tss < 1e-12: # Avoid division by zero if y_true is constant or zero
        # If y_true is constant, and y_pred is also that constant, press is 0, Q2 is 1.
        # If y_true is constant, and y_pred is different, press > 0, Q2 < 0.
        return 1.0 if press < 1e-12 else -float('inf')
    return float(1 - press / tss)


def _construct_milr_weight_tensor(
    core_shape: Tuple[int, ...],
    mode_ranks_for_core_modes: List[int], 
    alpha: float,
    num_loading_matrices_for_core: int, 
    dev: torch.device
) -> torch.Tensor:
    """
    Constructs the Mean Index-Weighted L2 regularization weight tensor W_G or W_D.
    core_shape: Shape of the core tensor (e.g., (1, L2, L3, ..., LN)).
    mode_ranks_for_core_modes: List of ranks [L2, L3, ..., LN] or [K2, ..., KM].
                               These are the L_{j+1} values from the formula.
    alpha: Weighting exponent.
    num_loading_matrices_for_core: N-1 for G, M-1 for D.
    """
    if num_loading_matrices_for_core == 0: 
        return torch.zeros(core_shape, device=dev, dtype=torch.float64)

    active_core_shape = core_shape[1:] # Modes corresponding to L2...LN or K2...KM
    
    if not active_core_shape: # Should only happen if core_shape is (1,) AND num_loading_matrices_for_core > 0 (inconsistent)
        return torch.zeros(core_shape, device=dev, dtype=torch.float64)

    # mesh_idx_grids will be a list of tensors, one for each active mode.
    # For a core (1,L2,L3), active_core_shape=(L2,L3).
    # mesh_idx_grids[0] will have shape (L2,L3) with values [[0,0,0],[1,1,1]] (for L2 index, if L2=2,L3=3)
    # mesh_idx_grids[1] will have shape (L2,L3) with values [[0,1,2],[0,1,2]] (for L3 index)
    try:
        mesh_idx_grids = torch.meshgrid(*[torch.arange(s, device=dev) for s in active_core_shape], indexing='ij')
    except RuntimeError as e: # Catches cases like empty active_core_shape for scalar with num_loading_matrices > 0
        if not active_core_shape and num_loading_matrices_for_core > 0:
             # This state indicates an issue, e.g. core_shape=(1,) but N-1 > 0
            return torch.zeros(core_shape, device=dev, dtype=torch.float64)
        raise e


    sum_normalized_indices_powered = torch.zeros(active_core_shape, device=dev, dtype=torch.float64)

    for i in range(num_loading_matrices_for_core):
        # i corresponds to the i-th loading matrix P^(i+1) or Q^(i+1) (0-indexed loop for P^(1)...P^(N-1))
        # The indices for this loading matrix's contribution are in mesh_idx_grids[i]
        # The rank (number of columns) for P^(i+1) is mode_ranks_for_core_modes[i]
        
        one_based_indices = mesh_idx_grids[i].double() + 1.0 # Convert 0-based torch index to 1-based for formula
        rank_L_for_current_mode = float(mode_ranks_for_core_modes[i])
        
        if rank_L_for_current_mode == 0: # Avoid division by zero if a rank is 0 (should not happen with valid Ln/Km)
            term = torch.zeros_like(one_based_indices)
        else:
            term = (one_based_indices / rank_L_for_current_mode) ** alpha
        sum_normalized_indices_powered += term
        
    weights_for_active_modes = (1.0 / num_loading_matrices_for_core) * sum_normalized_indices_powered
    
    final_weights = weights_for_active_modes.unsqueeze(0) # Add back the first mode of size 1
    
    return final_weights

class HOPLS_MILR:
    def __init__(
        self,
        R: int,
        Ln: List[int], 
        Km: Optional[List[int]] = None, 
        metric: Optional[callable] = None,
        epsilon: float = 1e-9, 
        lambda_X: float = 1e-3,
        lambda_Y: float = 1e-3,
        alpha: float = 1.0,
    ) -> None:
        self.R = R
        self.Ln_param = Ln 
        self.Km_param = Km 
        self.metric = metric or qsquared_score
        self.epsilon = epsilon
        self.lambda_X = float(lambda_X)
        self.lambda_Y = float(lambda_Y)
        self.alpha = float(alpha)

        self._is_matrix_Y: bool = False
        self.N_modal_X: int = 0 
        self.M_modal_Y: int = 0 
        
        self.actual_Ln_used: List[int] = []
        self.actual_Km_used: List[int] = [] 

        # Model components
        self.P_r_all_components: List[List[torch.Tensor]] = [] 
        self.Q_r_all_components: Union[List[List[torch.Tensor]], torch.Tensor] = []
        self.G_r_all_components: List[torch.Tensor] = [] 
        self.D_r_all_components: Union[List[torch.Tensor], torch.Tensor] = [] 
        self.T_mat: Optional[torch.Tensor] = None 
        self.W_mat: Optional[torch.Tensor] = None
        self.num_components_fitted = 0

    def _fit_tensor_X_matrix_Y(self, X: torch.Tensor, Y: torch.Tensor) -> "HOPLS_MILR":
        self._is_matrix_Y = True
        I1_samples = X.shape[0]
        M_responses_Y = Y.shape[1]
        
        self.N_modal_X = X.ndim - 1
        self.actual_Ln_used = list(self.Ln_param) # Ensure it's a list
        if len(self.actual_Ln_used) != self.N_modal_X and self.N_modal_X > 0 :
            raise ValueError(f"Length of Ln ({len(self.actual_Ln_used)}) must match X.ndim-1 ({self.N_modal_X})")

        Er, Fr_mat = X.clone(), Y.clone()

        P_r_list_of_lists_accum: List[List[torch.Tensor]] = []
        q_r_list_accum: List[torch.Tensor] = []
        G_r_list_accum: List[torch.Tensor] = []
        d_r_list_accum: List[torch.Tensor] = []
        t_r_list_accum: List[torch.Tensor] = []
        W_r_list_accum: List[torch.Tensor] = []
        
        num_fitted_for_loop = 0
        for _r_component_idx_not_used in range(self.R): # Use placeholder for loop var
            if torch.norm(Er) < self.epsilon or torch.norm(Fr_mat) < self.epsilon:
                break
            num_fitted_for_loop += 1

            Cr = mode_dot(Er, Fr_mat.T, mode=0) 
            
            ranks_for_Cr_HOOI = [1] + self.actual_Ln_used
            _Gr_C_dummy, factors_Cr = tucker(Cr, rank=ranks_for_Cr_HOOI, init="svd", n_iter_max=25, tol=1e-7)
            
            q_r = factors_Cr[0] 
            q_r_norm = torch.norm(q_r).clamp(min=self.epsilon)
            q_r = q_r / q_r_norm

            P_r_current_comp = []
            for p_factor in factors_Cr[1:]:
                # Normalize each column of p_factor
                col_norms = torch.norm(p_factor, dim=0, keepdim=True).clamp(min=self.epsilon)
                P_r_current_comp.append(p_factor / col_norms)

            _X_proj_for_tr = Er
            if self.N_modal_X > 0:
                _X_proj_for_tr = multi_mode_dot(Er, [p.T for p in P_r_current_comp], modes=list(range(1, self.N_modal_X + 1)))
            
            U_tr, _, _ = torch.linalg.svd(matricize_n(_X_proj_for_tr, mode=0), full_matrices=False)
            t_r = U_tr[:, 0:1] 
            t_r_norm = torch.norm(t_r).clamp(min=self.epsilon)
            t_r = t_r / t_r_norm
            
            t_r_list_accum.append(t_r)
            q_r_list_accum.append(q_r)
            P_r_list_of_lists_accum.append(P_r_current_comp)

            G_r_LS = mode_dot(_X_proj_for_tr, t_r.T, mode=0) 
            W_G_r = _construct_milr_weight_tensor(G_r_LS.shape, self.actual_Ln_used, self.alpha, self.N_modal_X, device)
            G_r = G_r_LS / (1.0 + self.lambda_X * W_G_r).clamp(min=self.epsilon)
            G_r_list_accum.append(G_r)

            u_r_vec = Fr_mat @ q_r 
            d_r_LS = (t_r.T @ u_r_vec).squeeze() 
            d_r = d_r_LS / (1.0 + self.lambda_Y) 
            d_r_list_accum.append(d_r.reshape(1)) 

            G_r_mat = matricize_n(G_r, mode=0) 
            if P_r_current_comp: 
                P_kron = kronecker(P_r_current_comp[::-1]) 
                norm_G_r_mat_sq = torch.norm(G_r_mat)**2
                G_r_vec_pinv = G_r_mat.T / norm_G_r_mat_sq.clamp(min=self.epsilon**2) if norm_G_r_mat_sq > self.epsilon**2 else torch.linalg.pinv(G_r_mat.T) # more robust pinv
                W_r_val = P_kron @ G_r_vec_pinv
            else: 
                g_val = G_r.item()
                W_r_val = torch.tensor([[1.0/g_val if abs(g_val) > self.epsilon else 0.0]], device=device, dtype=X.dtype)
            W_r_list_accum.append(W_r_val)
            
            Er = Er - tucker_to_tensor((G_r, [t_r] + P_r_current_comp))
            Fr_mat = Fr_mat - d_r * (t_r @ q_r.T)
        
        self.num_components_fitted = num_fitted_for_loop
        self.P_r_all_components = P_r_list_of_lists_accum
        self.Q_r_all_components = torch.cat(q_r_list_accum, dim=1) if q_r_list_accum else torch.empty((M_responses_Y, 0), device=device, dtype=X.dtype)
        self.G_r_all_components = G_r_list_accum
        self.D_r_all_components = torch.cat(d_r_list_accum) if d_r_list_accum else torch.empty(0, device=device, dtype=X.dtype)
        self.T_mat = torch.cat(t_r_list_accum, dim=1) if t_r_list_accum else torch.empty((I1_samples, 0), device=device, dtype=X.dtype)
        
        prod_X_non_sample_dims = X.shape[1:].numel() if X.ndim > 1 else (1 if X.ndim == 1 and X.shape[0] > 0 else 0) # Handle scalar X edge case
        if prod_X_non_sample_dims == 0 and X.ndim ==1 and X.shape[0] > 0 : prod_X_non_sample_dims = 1 # if X is (N_samples,)
        
        self.W_mat = torch.cat(W_r_list_accum, dim=1) if W_r_list_accum else torch.empty((prod_X_non_sample_dims, 0), device=device, dtype=X.dtype)
        
        return self

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> "HOPLS_MILR":
        X = X.to(device)
        Y = Y.to(device)

        if Y.ndim <= 1: # Ensure Y is at least a matrix
            Y = Y.unsqueeze(1) if Y.ndim == 1 else Y.reshape(Y.shape[0],1) # Make it N_samples x 1


        if Y.ndim == 2:
            return self._fit_tensor_X_matrix_Y(X, Y)
        
        self._is_matrix_Y = False
        I1_samples = X.shape[0]
        
        self.N_modal_X = X.ndim - 1
        self.M_modal_Y = Y.ndim - 1
        
        self.actual_Ln_used = list(self.Ln_param)
        self.actual_Km_used = list(self.Km_param) if self.Km_param is not None else ([1] * self.M_modal_Y if self.M_modal_Y > 0 else [])

        if len(self.actual_Ln_used) != self.N_modal_X and self.N_modal_X > 0:
            raise ValueError(f"Length of Ln ({len(self.actual_Ln_used)}) must match X.ndim-1 ({self.N_modal_X}) for tensor X")
        if len(self.actual_Km_used) != self.M_modal_Y and self.M_modal_Y > 0: # M_modal_Y can be 0 if Y is vector
            raise ValueError(f"Length of Km ({len(self.actual_Km_used)}) must match Y.ndim-1 ({self.M_modal_Y}) for tensor Y")

        Er, Fr = X.clone(), Y.clone()

        P_r_list_of_lists_accum: List[List[torch.Tensor]] = []
        Q_r_list_of_lists_accum: List[List[torch.Tensor]] = []
        G_r_list_accum: List[torch.Tensor] = []
        D_r_list_accum: List[torch.Tensor] = []
        t_r_list_accum: List[torch.Tensor] = []
        W_r_list_accum: List[torch.Tensor] = []

        num_fitted_for_loop = 0
        for _r_component_idx_not_used in range(self.R):
            if torch.norm(Er) < self.epsilon or torch.norm(Fr) < self.epsilon:
                break
            num_fitted_for_loop += 1
            
            Cr = torch.tensordot(Er, Fr, dims=([0], [0])) 
            
            ranks_for_Cr_HOOI = self.actual_Ln_used + self.actual_Km_used
            if not ranks_for_Cr_HOOI: 
                 _Gr_C_dummy = Cr # Cr is scalar
                 P_r_current_comp = []
                 Q_r_current_comp = []
            else:
                _Gr_C_dummy, factors_Cr = tucker(Cr, rank=ranks_for_Cr_HOOI, init="svd", n_iter_max=25, tol=1e-7)
                P_r_current_comp = factors_Cr[:self.N_modal_X]
                Q_r_current_comp = factors_Cr[self.N_modal_X:]
            
            temp_P_r_norm = []
            for p_factor in P_r_current_comp:
                col_norms = torch.norm(p_factor, dim=0, keepdim=True).clamp(min=self.epsilon)
                temp_P_r_norm.append(p_factor / col_norms)
            P_r_current_comp = temp_P_r_norm

            temp_Q_r_norm = []
            for q_factor in Q_r_current_comp:
                col_norms = torch.norm(q_factor, dim=0, keepdim=True).clamp(min=self.epsilon)
                temp_Q_r_norm.append(q_factor / col_norms)
            Q_r_current_comp = temp_Q_r_norm


            _X_proj_for_tr = Er
            if self.N_modal_X > 0:
                _X_proj_for_tr = multi_mode_dot(Er, [p.T for p in P_r_current_comp], modes=list(range(1, self.N_modal_X + 1)))
            
            U_tr, _, _ = torch.linalg.svd(matricize_n(_X_proj_for_tr, mode=0), full_matrices=False)
            t_r = U_tr[:, 0:1]
            t_r_norm = torch.norm(t_r).clamp(min=self.epsilon)
            t_r = t_r / t_r_norm

            t_r_list_accum.append(t_r)
            P_r_list_of_lists_accum.append(P_r_current_comp)
            Q_r_list_of_lists_accum.append(Q_r_current_comp)

            G_r_LS = mode_dot(_X_proj_for_tr, t_r.T, mode=0) 
            W_G_r = _construct_milr_weight_tensor(G_r_LS.shape, self.actual_Ln_used, self.alpha, self.N_modal_X, device)
            G_r = G_r_LS / (1.0 + self.lambda_X * W_G_r).clamp(min=self.epsilon)
            G_r_list_accum.append(G_r)

            _Y_proj_for_Dr = Fr
            if self.M_modal_Y > 0:
                _Y_proj_for_Dr = multi_mode_dot(Fr, [q.T for q in Q_r_current_comp], modes=list(range(1, self.M_modal_Y + 1)))
            D_r_LS = mode_dot(_Y_proj_for_Dr, t_r.T, mode=0) 
            W_D_r = _construct_milr_weight_tensor(D_r_LS.shape, self.actual_Km_used, self.alpha, self.M_modal_Y, device)
            D_r = D_r_LS / (1.0 + self.lambda_Y * W_D_r).clamp(min=self.epsilon)
            D_r_list_accum.append(D_r)
            
            G_r_mat = matricize_n(G_r, mode=0) 
            if P_r_current_comp:
                P_kron = kronecker(P_r_current_comp[::-1])
                norm_G_r_mat_sq = torch.norm(G_r_mat)**2
                G_r_vec_pinv = G_r_mat.T / norm_G_r_mat_sq.clamp(min=self.epsilon**2) if norm_G_r_mat_sq > self.epsilon**2 else torch.linalg.pinv(G_r_mat.T)
                W_r_val = P_kron @ G_r_vec_pinv
            else: 
                g_val = G_r.item()
                W_r_val = torch.tensor([[1.0/g_val if abs(g_val) > self.epsilon else 0.0]], device=device, dtype=X.dtype)
            W_r_list_accum.append(W_r_val)

            Er = Er - tucker_to_tensor((G_r, [t_r] + P_r_current_comp))
            Fr = Fr - tucker_to_tensor((D_r, [t_r] + Q_r_current_comp))
        
        self.num_components_fitted = num_fitted_for_loop
        self.P_r_all_components = P_r_list_of_lists_accum
        self.Q_r_all_components = Q_r_list_of_lists_accum
        self.G_r_all_components = G_r_list_accum
        self.D_r_all_components = D_r_list_accum
        self.T_mat = torch.cat(t_r_list_accum, dim=1) if t_r_list_accum else torch.empty((I1_samples, 0), device=device, dtype=X.dtype)
        
        prod_X_non_sample_dims = X.shape[1:].numel() if X.ndim > 1 else (1 if X.ndim == 1 and X.shape[0] > 0 else 0)
        if prod_X_non_sample_dims == 0 and X.ndim ==1 and X.shape[0] > 0 : prod_X_non_sample_dims = 1


        self.W_mat = torch.cat(W_r_list_accum, dim=1) if W_r_list_accum else torch.empty((prod_X_non_sample_dims, 0), device=device, dtype=X.dtype)

        return self
    
    def predict(
        self, X_new: torch.Tensor, Y_true_for_shape_and_metric: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, int, List[float]]:
        if self.T_mat is None or self.W_mat is None :
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        output_Y_shape = list(X_new.shape[:1]) 
        temp_Y_ndim_for_shape = 2 # Default to matrix if Y_true not provided and model context ambiguous
        if Y_true_for_shape_and_metric is not None:
            output_Y_shape.extend(Y_true_for_shape_and_metric.shape[1:])
            temp_Y_ndim_for_shape = Y_true_for_shape_and_metric.ndim
        elif self._is_matrix_Y:
            q_data = self.Q_r_all_components 
            output_Y_shape.append(q_data.shape[0] if isinstance(q_data, torch.Tensor) and q_data.ndim == 2 and q_data.shape[0] > 0 else 1)
            temp_Y_ndim_for_shape = 2
        else: # Tensor Y
            if self.D_r_all_components and isinstance(self.D_r_all_components, list) and self.D_r_all_components:
                 output_Y_shape.extend(self.D_r_all_components[0].shape[1:])
                 temp_Y_ndim_for_shape = len(output_Y_shape) # N_samples + M-1 modes
            else: output_Y_shape.append(1) 
        
        if temp_Y_ndim_for_shape <= 1 and len(output_Y_shape) == 1: # Ensure Y_pred is at least 2D (N_samples x 1)
            output_Y_shape.append(1)


        if self.num_components_fitted == 0: 
            return torch.zeros(output_Y_shape, device=device, dtype=X_new.dtype), 0, []

        X_new_dev = X_new.to(device)
        # Handle if X_new is a vector (N_samples,)
        if X_new_dev.ndim == 1: X_new_dev = X_new_dev.unsqueeze(1) # Treat as N_samples x 1
        X_new_mat = matricize_n(X_new_dev, mode=0) 
        
        if self.W_mat.shape[0] != X_new_mat.shape[1]: # W_mat rows should match X_new_mat cols
             # This can happen if X was vector (N_samples,) and W_mat became 1xR, but X_new is (N_new_samples, D_features)
             # Or if X during fit was (N_samples, D_features) and W_mat is (D_features, R), but X_new is (N_new_samples,)
             # Fallback or error: For now, assume a shape mismatch implies an issue with input or fit state for prediction.
             # Let's try to reshape W_mat if X_new_mat has 1 feature and W_mat has many, or vice versa for scalar fit.
            if X_new_mat.shape[1] == 1 and self.W_mat.shape[0] > 1: # X_new is effectively a vector of samples
                # This specific scenario is tricky. Assuming W_mat was built for a multi-feature X.
                # A common case is if X during fit was vector-like (N_samples,) -> W_mat (1, R_fitted)
                # And X_new is also (N_new_samples,) -> X_new_mat (N_new_samples, 1)
                # This should be fine. The problem arises if features mismatch.
                pass # Let matmul handle if dimensions are compatible (e.g. (N,1) @ (1,R))
            elif self.W_mat.shape[0] == 1 and X_new_mat.shape[1] > 1 : # W_mat from vector X, X_new has features
                 # This is an error - cannot apply weights from single feature model to multi-feature input
                 raise ValueError(f"W_mat feature dim ({self.W_mat.shape[0]}) != X_new feature dim ({X_new_mat.shape[1]}). Model likely fit on vector X.")
            # else: pass, let matmul throw error if incompatible


        T_new_all_r_mat = X_new_mat @ self.W_mat 

        best_q2_val = -float('inf')
        best_Y_pred_val: Optional[torch.Tensor] = None
        best_r_val = 0
        q2s_list: List[float] = []

        Y_true_dev: Optional[torch.Tensor] = None
        if Y_true_for_shape_and_metric is not None:
            Y_true_dev = Y_true_for_shape_and_metric.to(device)
            if Y_true_dev.ndim <=1: Y_true_dev = Y_true_dev.unsqueeze(1) if Y_true_dev.ndim == 1 else Y_true_dev.reshape(Y_true_dev.shape[0],1)


        for r_to_use in range(1, self.num_components_fitted + 1):
            T_new_current_r_iter = T_new_all_r_mat[:, :r_to_use] 

            if self._is_matrix_Y:
                Q_r_subset = self.Q_r_all_components[:, :r_to_use] 
                D_r_diag_subset = torch.diag(self.D_r_all_components[:r_to_use]) 
                Y_pred_current_iter = T_new_current_r_iter @ D_r_diag_subset @ Q_r_subset.T
            else: # Tensor Y
                Y_pred_current_iter = torch.zeros(output_Y_shape, device=device, dtype=X_new.dtype)
                for i_comp_idx in range(r_to_use):
                    t_new_for_comp = T_new_current_r_iter[:, i_comp_idx:i_comp_idx+1]
                    D_comp = self.D_r_all_components[i_comp_idx]            
                    Q_comp_list = self.Q_r_all_components[i_comp_idx]       
                    term_Y_pred = tucker_to_tensor((D_comp, [t_new_for_comp] + Q_comp_list))
                    Y_pred_current_iter = Y_pred_current_iter + term_Y_pred
            
            if Y_true_dev is not None and Y_pred_current_iter.numel() > 0 and Y_true_dev.numel() > 0 :
                if Y_pred_current_iter.shape != Y_true_dev.shape: # Ensure shapes match for metric
                    # This might happen if output_Y_shape logic had a default that didn't match Y_true_dev
                    # Attempt to reshape Y_pred if Y_true_dev is the source of truth for shape
                     try:
                        Y_pred_current_iter = Y_pred_current_iter.reshape_as(Y_true_dev)
                     except RuntimeError: # If reshape is not possible
                        q2_val = -float('inf') # Penalize shape mismatch severely
                
                q2_val = self.metric(Y_true_dev, Y_pred_current_iter)
                q2s_list.append(q2_val)
                if q2_val > best_q2_val:
                    best_q2_val = q2_val
                    best_Y_pred_val = Y_pred_current_iter
                    best_r_val = r_to_use
            elif r_to_use == self.num_components_fitted and Y_true_dev is None:
                best_Y_pred_val = Y_pred_current_iter 
                best_r_val = r_to_use
        
        if best_Y_pred_val is None: 
            if self.num_components_fitted > 0 and 'Y_pred_current_iter' in locals() and Y_pred_current_iter.numel() > 0: 
                best_Y_pred_val = Y_pred_current_iter 
                best_r_val = self.num_components_fitted
            else: 
                 best_Y_pred_val = torch.zeros(output_Y_shape, device=device, dtype=X_new.dtype)
                 best_r_val = 0
        
        return best_Y_pred_val, best_r_val, q2s_list

    def score(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        if self.T_mat is None: 
             self.fit(X,Y)
        
        _Y_pred, _best_r, q2s = self.predict(X, Y_true_for_shape_and_metric=Y)
        if not q2s:
            return -float('inf') 
        return max(q2s)