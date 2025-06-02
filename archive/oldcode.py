
# def matricize(data: torch.Tensor) -> torch.Tensor:
#     """
#     Unfold a tensor into a matrix along mode-1 (axis 0), 
#     using Fortran order so that fibres are contiguous.
#     E.g. a 4×3×2 tensor → 4×6 matrix.
#     """
#     flat = np.reshape(
#         data.detach().cpu().numpy(),
#         (-1, np.prod(data.shape[1:])),
#         order="F"
#     )
#     return torch.from_numpy(flat).to(data.dtype)


# def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
#     """Compute Root Mean Square Error between two arrays."""
#     return float(np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0)))


# def qsquared(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
#     """
#     Compute the Q² statistic (1 – PRESS/TSS) for two tensors/matrices.
#     Used as the default metric in HOPLS.
#     """
#     return float(
#         1 - (torch.norm(y_true - y_pred) ** 2) / (torch.norm(y_true) ** 2)
#     )
    


# Tensor Contraction Formula across mode (1-indexing):
# C[i1,...,in-1,in+1,...iN, j1,...,jm-1, jm+1,...jm ] = 
# ∑{in=1..In} A[i1,...,in-1,in, in+1,...iN]*B[j1,...,jm-1, jm+1,...jm ]

# def cov(A, B):
#     """Computes the mode 1 (mode 0 in python) contraction of 2 matrices."""
#     assert A.shape[0] == B.shape[0], "A and B need to have the same shape on axis 0"
#     dimension_A = A.shape[1:]
#     dimension_B = B.shape[1:]
#     dimensions = list(dimension_A) + list(dimension_B)
#     rmode_A = len(dimension_A)
#     dim = A.shape[0]
#     C = tl.zeros(dimensions)
#     indices = []
#     for mode in dimensions:
#         indices.append(range(mode))
#     for idx in product(*indices):
#         idx_A, idx_B = list(idx[:rmode_A]), list(idx[rmode_A:])
#         C[idx] = np.sum(
#             [A[tuple([i] + idx_A)] * B[tuple([i] + idx_B)] for i in range(dim)]
#         )
#     return C



# class HOPLS:
#     def __init__(
#         self,
#         R: int,
#         Ln: List[int],
#         Km: Optional[List[int]] = None,
#         metric: Optional[callable] = None,
#         epsilon: float = 1e-6,
#     ) -> None:
#         """
#         Higher-Order Partial Least Squares (HOPLS).

#         Parameters:
#             R       : int             – number of latent components.
#             Ln      : list of int     – Tucker ranks for modes 2…N of X.
#             Km      : list of int     – Tucker ranks for modes 2…M of Y.
#             metric  : callable        – scoring metric (default: qsquared).
#             epsilon : float           – deflation stopping threshold.
#         """
#         self.R       = R
#         self.Ln      = Ln
#         self.Km      = Km if Km is not None else [Ln[0]]
#         self.metric  = metric or qsquared
#         self.epsilon = epsilon

#         # number of non-sample modes
#         self.N = len(self.Ln)
#         self.M = len(self.Km)

#         # will hold (P, Q, D_list, T, W) after fit
#         self.model: Optional[Tuple] = None

#     def _fit_2d(self, X: torch.Tensor, Y: torch.Tensor) -> "HOPLS":
#         """
#         Special case when Y is a matrix (mode-2 tensor): Algorithm 2 in Zhao et al. (2012).
#         """
#         Er, Fr = X.clone(), Y.clone()
#         P: List[List[torch.Tensor]] = []
#         Q: List[torch.Tensor]       = []
#         D = torch.zeros((self.R, self.R), dtype=X.dtype)
#         T: List[torch.Tensor] = []
#         W: List[torch.Tensor] = []

#         for r in range(self.R):
#             # 1. check stopping criterion (||Er||, ||Fr|| > ε)
#             if torch.norm(Er) <= self.epsilon or torch.norm(Fr) <= self.epsilon:
#                 break

#             # 2. form covariance tensor C_r = mode_dot(Er, Frᵀ, mode=0)
#             Cr = mode_dot(Er, Fr.t(), mode=0)
            
#             # 3. perform HOOI‐based Tucker of C_r with ranks [1] + Ln  
#             #    → core GrC and loading factors [q_r, P_r¹,…,P_rᴺ]
#             GrC, latent_factors = tucker(Cr, rank=[1] + self.Ln)
#             # normalise 
#             q_r = latent_factors[0] / torch.norm(latent_factors[0])
#             P_r = [p / torch.norm(p) for p in latent_factors[1:]]

#             # 4. compute latent vector t_r by projecting Er onto P_r loadings:
#             #    t̃ = Er ×ₙ P_r⁽ᵀ⁾, then t_r = matricize(t̃)·pinv(mat(GrC)), normalised
#             t_tilde = multi_mode_dot(
#                 Er, P_r, modes=list(range(1, self.N+1)), transpose=True
#             )
#             GrC_pinv = torch.pinverse(matricize(GrC))
#             t_r = torch.mm(matricize(t_tilde), GrC_pinv)
#             t_r = t_r / torch.norm(t_r)

#             # 5. scalar regression from t_r to Y:  u_r = Fr·q_r,  d_r = u_rᵀ·t_r
#             u_r = torch.mm(Fr, q_r)
#             d_r = torch.mm(u_r.t(), t_r)
#             D[r, r] = d_r

#             # 6. build W_r = kron(P_r^(N)…P_r^(1)) · pinv(mat( G_d ) )
#             #    where G_d = Er ×₁ t_rᵀ ×₂ P_r^(1)ᵀ … ×ₙ P_r^(N)ᵀ
#             G_d = tucker_to_tensor((Er, [t_r] + P_r), transpose_factors=True)
#             G_d_pinv = torch.pinverse(matricize(G_d))
#             W_r = torch.mm(kronecker(P_r[::-1]), G_d_pinv)

#             # 7. deflate: Er ← Er – G_d ×₁ t_r ×₂ P_r¹ … ×ₙ P_rᴺ
#             #            Fr ← Fr – d_r·t_r·q_rᵀ
#             Er = Er - tucker_to_tensor((G_d, [t_r] + P_r))
#             Fr = Fr - d_r * torch.mm(t_r, q_r.t())

#             # collect
#             P.append(P_r)
#             Q.append(q_r)
#             T.append(t_r)
#             W.append(W_r)

#         # stack into final matrices
#         Q_mat = torch.cat(Q, dim=1)
#         T_mat = torch.cat(T, dim=1)
#         W_mat = torch.cat(W, dim=1)

#         self.model = (P, Q_mat, D, T_mat, W_mat)
#         return self

#     def fit(self, X: torch.Tensor, Y: torch.Tensor) -> "HOPLS":
#         """
#         Full HOPLS fit for X.ndim≥3, Y.ndim≥2 (Algorithm 1).
#         """
#         assert X.ndim >= 3, "X must be at least 3-way."
#         assert Y.ndim >= 2, "Y must be at least matrix (2-way)."
#         assert len(self.Ln) == X.ndim - 1

#         # if Y is 2-way, use the special routine
#         if Y.ndim == 2:
#             return self._fit_2d(X, Y)

#         assert len(self.Km) == Y.ndim - 1, f"The list of ranks for the decomposition of Y (Km) need to be equal to the mode of Y -1: {Y_mode-1}."


#         Er, Fr = X.clone(), Y.clone()
#         P: List[List[torch.Tensor]]      = []
#         Q: List[List[torch.Tensor]]      = []
#         G_list: List[torch.Tensor]       = []
#         D_list: List[torch.Tensor]       = []
#         T: List[torch.Tensor]            = []
#         W: List[torch.Tensor]            = []

#         for _ in range(self.R):
#             if torch.norm(Er) <= self.epsilon or torch.norm(Fr) <= self.epsilon:
#                 break

#             # 1. form full cross-covariance C_r = ∑ₖ Er[...,k]·Fr[...,k]
#             Cr = torch.Tensor(np.tensordot(Er, Fr, axes=(0,0)))

#              # 2. Tucker‐HOOI on C_r with ranks Ln+Km → factors for X‐side (Pr) and Y‐side (Qr)
#             _, latent_factors = tucker(Cr, rank=self.Ln + self.Km)
#             Pr = latent_factors[:self.N]
#             Qr = latent_factors[self.N:]

#             # 3. compute t_r = leading left SVD vector of Er ×ₙ P_r^(n)ᵀ
#             Er_proj = multi_mode_dot(
#                 Er, Pr, modes=list(range(1, self.N+1)), transpose=True
#             )
#             U, _, _ = torch.svd(matricize(Er_proj)) # left singular matrix
#             t_r = U[:, :1]

#             # 4. core tensors: 
#             #    G_r = Er ×₁ t_rᵀ ×₂ Pr¹ᵀ ×₃ ... ×ₙ Pr^(N-1)ᵀ
#             #    D_r = Fr ×₁ t_rᵀ ×₂ Qr¹ᵀ ×₃ ... ×ₘ Qr^(M-1)ᵀ
#             G_r = tucker_to_tensor((Er, [t_r] + Pr), transpose_factors=True)
#             D_r = tucker_to_tensor((Fr, [t_r] + Qr), transpose_factors=True)

#             # 5. compute W_r = kron(Pr[::-1]) · pinv( mat(G_r) )
#             G_r_inv = torch.pinverse(matricize(G_r))
#             W_r = torch.mm(kronecker(Pr[::-1]), G_r_inv)

#             # 6. deflate both Er and Fr
#             Er = Er - tucker_to_tensor((G_r,[t_r] + Pr))
#             Fr = Fr - tucker_to_tensor((D_r,[t_r] + Qr))

#             # 7.collect
#             P.append(Pr)
#             Q.append(Qr)
#             G_list.append(G_r)
#             D_list.append(D_r)
#             T.append(t_r)
#             W.append(W_r)

#         # stack T and W
#         T_mat = torch.cat(T, dim=1)
#         W_mat = torch.cat(W, dim=1)
#         self.model = (P, Q, D_list, T_mat, W_mat)
#         return self

#     def predict(self, X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, int]:
#         """Compute the HOPLS for X and Y with respect to the parameters R, Ln and Km.

#         Parameters:
#             X: tensorly Tensor, The tensor we wish to do a prediction from.
#             Of shape [i1, ... iN], N >= 3.

#             Y: tensorly Tensor, used only for the shape of the prediction.

#         Returns:
#             Y_pred: tensorly Tensor, The predicted Y from the model.
#         """
#         # Unpack model parameters: Q holds loadings for Y, D holds diagonal matrices, W holds weights for X
#         _, Q, D, _, W = self.model

#         # Initialise best Q² statistic to negative infinity for comparison
#         best_q2 = -np.inf

#         # If Y is a higher-order tensor (more than 2 modes), build the matricised Q*
#         if len(Y.shape) > 2:
#             Q_star = []
#             # Loop over each latent component
#             for r in range(self.R):
#                 # Build the Kronecker product of the r-th factor matrices of Q
#                 Qkron = kronecker([Q[r][self.M - m - 1] for m in range(self.M)])
#                 # Matricise D[r] and multiply by the transpose of the Kron product
#                 Q_star.append(torch.mm(
#                     matricize(D[r][np.newaxis, ...]),
#                     Qkron.t()
#                 ))
#             # Concatenate all components into one projection matrix
#             Q_star = torch.cat(Q_star)

#         # List to store Q² values for each number of components
#         q2s = []

#         # Evaluate prediction for 1..R components to find optimal R
#         for r in range(1, self.R + 1):
#             # For the special case of a two-way Y (matrix), compute Q* directly
#             if len(Y.shape) == 2:
#                 # Use first r×r block of D and first r columns of Q
#                 Q_star = torch.mm(D[:r, :r], Q[:, :r].t())

#             # Compute the inner projection: X-weights times Y-loadings
#             inter = torch.mm(W[:, :r], Q_star[:r])

#             # Matricise X, project into Y-space, then reshape back to Y's original shape (Fortran order)
#             Y_pred = np.reshape(
#                 torch.mm(matricize(X), inter),
#                 Y.shape,
#                 order="F"
#             )

#             # Compute the Q² statistic between true and predicted Y
#             Q2 = qsquared(Y, Y_pred)
#             q2s.append(Q2)

#             # Update the best model if this Q² is higher
#             if Q2 > best_q2:
#                 best_q2 = Q2
#                 best_r = r
#                 best_Y_pred = Y_pred

#         # Return the prediction corresponding to the optimal number of components
#         return best_Y_pred, best_r, q2s
    
#     def score(self, X: torch.Tensor, Y: torch.Tensor) -> float:
#         """Fit + predict and return the chosen metric."""
#         self.fit(X, Y)
#         Y_pred, _ = self.predict(X, Y)
#         return self.metric(
#             Y.reshape(Y.shape[0], -1),
#             Y_pred.reshape(Y_pred.shape[0], -1)
#         )
