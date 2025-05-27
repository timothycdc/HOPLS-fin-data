from typing import List, Optional, Tuple

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
        self.Km = Km if Km is not None else [Ln[0]]
        self.metric = metric or qsquared
        self.epsilon = epsilon

        # number of non-sample modes
        self.N = len(self.Ln)
        self.M = len(self.Km)

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
                Er, P_r, modes=list(range(1, self.N + 1)), transpose=True
            )
            # NEW: use torch.linalg.pinv on device
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
        """Compute the HOPLS for X and Y with respect to the parameters R, Ln and Km."""
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
            warnings.warn(
                "HOPLS.predict: no optimal component found, using last Y_pred"
            )
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
                Er, P_r, modes=list(range(1, self.N + 1)), transpose=True
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
