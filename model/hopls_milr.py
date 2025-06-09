
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
        Ln: Union[List[int], Tuple[int]],
        Km: Optional[List[int]] = None, 
        metric: Optional[callable] = None,
        epsilon: float = 1e-9, 
        lambda_X: float = 1e-3,
        lambda_Y: float = 1e-3,
        alpha: float = 1.0,
    ) -> None:
        self.R = R
        self.Ln = list(Ln)
        self.Km = Km 
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
        self.actual_Ln_used = list(self.Ln) # Ensure it's a list
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
        
        self.actual_Ln_used = list(self.Ln)
        self.actual_Km_used = list(self.Km) if self.Km is not None else ([1] * self.M_modal_Y if self.M_modal_Y > 0 else [])

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