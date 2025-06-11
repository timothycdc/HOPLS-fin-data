import torch
import tensorly as tl
from tensorly import tucker_to_tensor
from tensorly.decomposition import tucker
from tensorly.tenalg import kronecker, mode_dot, multi_mode_dot
from typing import List, Optional, Tuple, Union

# Ensure PyTorch backend is used for TensorLy
tl.set_backend("pytorch")

def matricize_n(tensor: torch.Tensor, mode: int = 0) -> torch.Tensor:
    """Unfolds a tensor into a matrix along a specified mode."""
    return tl.unfold(tensor, mode)

def qsquared_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Computes Q² score."""
    if y_true.shape != y_pred.shape:
        # Allow for broadcasting if one of them has trailing dimensions of 1
        try:
            y_true_b, y_pred_b = torch.broadcast_tensors(y_true, y_pred)
        except RuntimeError:
            raise ValueError(f"Shape mismatch and not broadcastable: y_true {y_true.shape}, y_pred {y_pred.shape}")
    else:
        y_true_b, y_pred_b = y_true, y_pred
            
    y_true_flat = y_true_b.contiguous().view(y_true_b.shape[0], -1)
    y_pred_flat = y_pred_b.contiguous().view(y_pred_b.shape[0], -1)
    
    press = torch.norm(y_true_flat - y_pred_flat) ** 2
    tss = torch.norm(y_true_flat - torch.mean(y_true_flat, dim=0, keepdim=True)) ** 2

    if tss.abs() < 1e-12: 
        return 1.0 if press.abs() < 1e-12 else -float('inf')
    q2 = 1 - press / tss
    return float(q2)

class HOPLS_RIDGE:
    def __init__(
        self,
        R: int,
        Ln: Union[List[int], Tuple[int]], 
        Km: Optional[List[int]] = None, 
        metric: Optional[callable] = None,
        epsilon: float = 1e-9, 
        lambda_X: float = 1e-3,
        lambda_Y: float = 1e-3
    ) -> None:
        self.R = R
        self.Ln = list(Ln) # Ensure Ln is a list
        self.Km = list(Km) if Km is not None else None # Ensure Km is a list if provided
        self.metric = metric or qsquared_score
        self.epsilon = epsilon
        
        if lambda_X < 0:
            raise ValueError("lambda_X (Ridge parameter for X) must be non-negative.")
        if lambda_Y < 0:
            raise ValueError("lambda_Y (Ridge parameter for Y) must be non-negative.")
            
        self.lambda_X = float(lambda_X)
        self.lambda_Y = float(lambda_Y)

        self._is_matrix_Y: bool = False
        self.N_modal_X: int = 0 
        self.M_modal_Y: int = 0 
        
        self.actual_Ln_used: List[int] = []
        self.actual_Km_used: List[int] = [] 

        self.P_r_all_components: List[List[torch.Tensor]] = [] 
        self.Q_r_all_components: Union[List[List[torch.Tensor]], torch.Tensor] = []
        self.G_r_all_components: List[torch.Tensor] = [] 
        self.D_r_all_components: Union[List[torch.Tensor], torch.Tensor] = [] 
        self.T_mat: Optional[torch.Tensor] = None 
        self.W_mat: Optional[torch.Tensor] = None # Projection weights X_mat @ W_mat = T_mat
        self.num_components_fitted = 0

    def _calculate_projection_weights_w_r(
        self,
        G_r: torch.Tensor,
        P_r_current_comp: List[torch.Tensor],
        current_device: torch.device,
        current_dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Calculates w_r^* = (P_r^(N-1) kron ... kron P_r^(1)) (vec(G_r))^dagger_transposed
        (vec(G_r))^dagger_transposed = vec(G_r) / ||vec(G_r)||^2
        """
        G_r_mat = matricize_n(G_r, mode=0) # Shape (1, K_core = L2*...*LN)
        
        # (vec(G_r))^dagger_transposed is G_r_mat.T / ||G_r_mat.T||^2
        # torch.linalg.pinv(G_r_mat) computes G_r_mat.T / ||G_r_mat||^2, which is (K_core, 1)
        g_pinv_col = torch.linalg.pinv(G_r_mat) # Shape (K_core, 1)

        if P_r_current_comp: 
            # P_r_current_comp is [P^(1)_r, ..., P^(N-1)_r]
            # Paper requires P_kron = P^(N-1)_r kron ... kron P^(1)_r
            # tensorly.tenalg.kronecker([A,B,C]) gives A tensor B tensor C
            # So we need to reverse P_r_current_comp for kronecker
            P_kron = kronecker(P_r_current_comp[::-1]) 
            W_r_val = P_kron @ g_pinv_col
        else: # N_modal_X = 0 (X is a vector), P_kron is effectively scalar 1
              # G_r is scalar, G_r_mat is (1,1), g_pinv_col is (1,1)
            W_r_val = g_pinv_col 
        return W_r_val

    def _fit_tensor_X_matrix_Y(self, X: torch.Tensor, Y: torch.Tensor) -> "HOPLS_RIDGE":
        self._is_matrix_Y = True
        I1_samples = X.shape[0]
        M_responses_Y = Y.shape[1]
        
        current_device = X.device
        current_dtype = X.dtype

        self.N_modal_X = X.ndim - 1
        self.actual_Ln_used = list(self.Ln)
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
        for _r_idx in range(self.R):
            if torch.norm(Er) < self.epsilon or torch.norm(Fr_mat) < self.epsilon:
                break
            num_fitted_for_loop += 1

            # Step 1: Find Loadings (on Cr)
            Cr = mode_dot(Er, Fr_mat.T, mode=0) # Shape (M_Y, I2, ..., IN)
            ranks_for_Cr_HOOI = [1] + self.actual_Ln_used # Rank 1 for Y-mode, Ln for X-modes
            
            _Gr_C_dummy, factors_Cr = tucker(Cr, rank=ranks_for_Cr_HOOI, init="svd", n_iter_max=50, tol=1e-8)
            
            q_r = factors_Cr[0] # Shape (M_Y, 1)
            q_r_norm = torch.norm(q_r).clamp(min=self.epsilon)
            q_r = q_r / q_r_norm

            P_r_current_comp = []
            if self.N_modal_X > 0: # factors_Cr[1:] are P_X factors
                for p_factor in factors_Cr[1:]:
                    # Orthonormalize (Tucker does not guarantee this for factors if ranks are smaller than full)
                    # For HOOI, factors should be column-orthonormal.
                    # However, to be safe and match typical PLS steps where loadings are normalized:
                    U_p, _, Vh_p = torch.linalg.svd(p_factor, full_matrices=False)
                    # Use U_p if p_factor columns >= rows, else Vh_p.T. Tucker should give tall matrices.
                    # For Tucker factors (I_k x L_k), they should be column orthonormal.
                    # A simple normalization of columns:
                    col_norms = torch.norm(p_factor, dim=0, keepdim=True).clamp(min=self.epsilon)
                    P_r_current_comp.append(p_factor / col_norms)


            # Step 2: Find Latent Vector t_r (from Er)
            _X_proj_for_tr = Er
            if self.N_modal_X > 0:
                _X_proj_for_tr = multi_mode_dot(Er, [p.T for p in P_r_current_comp], modes=list(range(1, self.N_modal_X + 1)))
            
            # _X_proj_for_tr has shape (I1, L2, ..., LN)
            # We need SVD of its mode-0 unfolding: (I1, L2*...*LN)
            U_tr, _, _ = torch.linalg.svd(matricize_n(_X_proj_for_tr, mode=0), full_matrices=False)
            t_r = U_tr[:, 0:1] # Shape (I1, 1)
            t_r_norm = torch.norm(t_r).clamp(min=self.epsilon)
            t_r = t_r / t_r_norm
            
            t_r_list_accum.append(t_r)
            q_r_list_accum.append(q_r)
            P_r_list_of_lists_accum.append(P_r_current_comp)

            # Step 3: Find Ridge Core G_r for X
            G_r_LS = mode_dot(_X_proj_for_tr, t_r.T, mode=0) # Shape (1, L2, ..., LN)
            G_r = G_r_LS / (1.0 + self.lambda_X) 
            G_r_list_accum.append(G_r)

            # Step 3b: Find Ridge scalar d_r for Y
            u_r_vec = Fr_mat @ q_r # Project Y onto q_r, result (I1, 1)
            d_r_LS = (t_r.T @ u_r_vec).squeeze() # Scalar
            d_r = d_r_LS / (1.0 + self.lambda_Y) 
            d_r_list_accum.append(d_r.reshape(1)) # Store as (1,) tensor

            # Calculate w_r^* for X block
            W_r_val = self._calculate_projection_weights_w_r(G_r, P_r_current_comp, current_device, current_dtype)
            W_r_list_accum.append(W_r_val)
            
            # Step 4: Deflation
            factors_for_Er_approx = [t_r] + P_r_current_comp
            Er_approx_r = tucker_to_tensor((G_r, factors_for_Er_approx))
            Er = Er - Er_approx_r
            Fr_mat = Fr_mat - d_r * (t_r @ q_r.T)
        
        self.num_components_fitted = num_fitted_for_loop
        self.P_r_all_components = P_r_list_of_lists_accum
        self.Q_r_all_components = torch.cat(q_r_list_accum, dim=1) if q_r_list_accum else torch.empty((M_responses_Y, 0), device=current_device, dtype=current_dtype)
        self.G_r_all_components = G_r_list_accum
        self.D_r_all_components = torch.cat(d_r_list_accum) if d_r_list_accum else torch.empty(0, device=current_device, dtype=current_dtype)
        self.T_mat = torch.cat(t_r_list_accum, dim=1) if t_r_list_accum else torch.empty((I1_samples, 0), device=current_device, dtype=current_dtype)
        
        # Determine shape for W_mat: (prod(I2,...,IN), R_fitted)
        # If X is (I1,), prod_X_non_sample_dims = 1
        prod_X_non_sample_dims = X.shape[1:].numel() if X.ndim > 1 else 1
        self.W_mat = torch.cat(W_r_list_accum, dim=1) if W_r_list_accum else torch.empty((prod_X_non_sample_dims, 0), device=current_device, dtype=current_dtype)
        
        return self

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> "HOPLS_RIDGE":
        if Y.ndim == 1: # Ensure Y is at least 2D (N_samples x N_responses)
            Y = Y.unsqueeze(1)
        elif Y.ndim == 0:
             raise ValueError("Y cannot be a scalar.")

        current_device = X.device
        current_dtype = X.dtype

        if Y.ndim == 2: # Y is a matrix
            return self._fit_tensor_X_matrix_Y(X, Y)
        
        # Y is a Tensor (>=3D)
        self._is_matrix_Y = False
        I1_samples = X.shape[0]
        
        self.N_modal_X = X.ndim - 1
        self.M_modal_Y = Y.ndim - 1 # Number of non-sample modes in Y
        
        self.actual_Ln_used = list(self.Ln)
        if self.Km is None : # Default Km if not provided for Tensor Y
             self.actual_Km_used = [1] * self.M_modal_Y if self.M_modal_Y > 0 else []
        else:
            self.actual_Km_used = list(self.Km)


        if len(self.actual_Ln_used) != self.N_modal_X and self.N_modal_X > 0:
            raise ValueError(f"Length of Ln ({len(self.actual_Ln_used)}) must match X.ndim-1 ({self.N_modal_X}) for tensor X")
        if len(self.actual_Km_used) != self.M_modal_Y and self.M_modal_Y > 0:
            raise ValueError(f"Length of Km ({len(self.actual_Km_used)}) must match Y.ndim-1 ({self.M_modal_Y}) for tensor Y")

        Er, Fr = X.clone(), Y.clone()

        P_r_list_of_lists_accum: List[List[torch.Tensor]] = []
        Q_r_list_of_lists_accum: List[List[torch.Tensor]] = []
        G_r_list_accum: List[torch.Tensor] = []
        D_r_list_accum: List[torch.Tensor] = []
        t_r_list_accum: List[torch.Tensor] = []
        W_r_list_accum: List[torch.Tensor] = []

        num_fitted_for_loop = 0
        for _r_idx in range(self.R):
            if torch.norm(Er) < self.epsilon or torch.norm(Fr) < self.epsilon:
                break
            num_fitted_for_loop += 1
            
            # Step 1: Find Loadings (on Cr)
            # Cr = <Er, Fr>_{1;1}
            Cr = torch.tensordot(Er, Fr, dims=([0], [0])) # Shape (I2..IN, J2..JM)
            ranks_for_Cr_HOOI = self.actual_Ln_used + self.actual_Km_used
            
            if not ranks_for_Cr_HOOI: # Both X and Y are effectively vectors (after sample mode)
                 _Gr_C_dummy = Cr 
                 factors_Cr = []
            else:
                _Gr_C_dummy, factors_Cr = tucker(Cr, rank=ranks_for_Cr_HOOI, init="svd", n_iter_max=50, tol=1e-8)

            P_r_factors = factors_Cr[:self.N_modal_X]
            Q_r_factors = factors_Cr[self.N_modal_X:]
            
            P_r_current_comp = []
            for p_factor in P_r_factors:
                col_norms = torch.norm(p_factor, dim=0, keepdim=True).clamp(min=self.epsilon)
                P_r_current_comp.append(p_factor / col_norms)

            Q_r_current_comp = []
            for q_factor in Q_r_factors:
                col_norms = torch.norm(q_factor, dim=0, keepdim=True).clamp(min=self.epsilon)
                Q_r_current_comp.append(q_factor / col_norms)

            # Step 2: Find Latent Vector t_r (from Er)
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

            # Step 3: Find Ridge Core G_r for X
            G_r_LS = mode_dot(_X_proj_for_tr, t_r.T, mode=0) 
            G_r = G_r_LS / (1.0 + self.lambda_X) 
            G_r_list_accum.append(G_r)

            # Step 3b: Find Ridge Core D_r for Y
            _Y_proj_for_Dr = Fr
            if self.M_modal_Y > 0:
                _Y_proj_for_Dr = multi_mode_dot(Fr, [q.T for q in Q_r_current_comp], modes=list(range(1, self.M_modal_Y + 1)))
            D_r_LS = mode_dot(_Y_proj_for_Dr, t_r.T, mode=0) 
            D_r = D_r_LS / (1.0 + self.lambda_Y) 
            D_r_list_accum.append(D_r)
            
            # Calculate w_r^* for X block
            W_r_val = self._calculate_projection_weights_w_r(G_r, P_r_current_comp, current_device, current_dtype)
            W_r_list_accum.append(W_r_val)

            # Step 4: Deflation
            factors_for_Er_approx = [t_r] + P_r_current_comp
            Er_approx_r = tucker_to_tensor((G_r, factors_for_Er_approx))
            Er = Er - Er_approx_r
            
            factors_for_Fr_approx = [t_r] + Q_r_current_comp
            Fr_approx_r = tucker_to_tensor((D_r, factors_for_Fr_approx))
            Fr = Fr - Fr_approx_r
        
        self.num_components_fitted = num_fitted_for_loop
        self.P_r_all_components = P_r_list_of_lists_accum
        self.Q_r_all_components = Q_r_list_of_lists_accum # List[List[torch.Tensor]]
        self.G_r_all_components = G_r_list_accum
        self.D_r_all_components = D_r_list_accum # List[torch.Tensor]
        self.T_mat = torch.cat(t_r_list_accum, dim=1) if t_r_list_accum else torch.empty((I1_samples, 0), device=current_device, dtype=current_dtype)
        
        prod_X_non_sample_dims = X.shape[1:].numel() if X.ndim > 1 else 1
        self.W_mat = torch.cat(W_r_list_accum, dim=1) if W_r_list_accum else torch.empty((prod_X_non_sample_dims, 0), device=current_device, dtype=current_dtype)

        return self
    
    def _determine_prediction_shape(self, X_new: torch.Tensor, Y_true_for_shape_and_metric: Optional[torch.Tensor]) -> torch.Size:
        output_Y_shape_dims = [X_new.shape[0]] # N_samples
        if Y_true_for_shape_and_metric is not None:
            output_Y_shape_dims.extend(list(Y_true_for_shape_and_metric.shape[1:]))
        elif self._is_matrix_Y:
            q_data = self.Q_r_all_components # This is a matrix (M_responses_Y, R_fitted)
            if isinstance(q_data, torch.Tensor) and q_data.ndim == 2 and q_data.numel() > 0:
                output_Y_shape_dims.append(q_data.shape[0]) # M_responses_Y
            else: # Fallback if Q_r_all_components is empty or not as expected
                output_Y_shape_dims.append(1) 
        else: # Tensor Y
            # Try to infer J_m dimensions from the stored Q_r factors of the first component
            if self.Q_r_all_components and isinstance(self.Q_r_all_components, list) and \
               self.Q_r_all_components[0] and isinstance(self.Q_r_all_components[0], list) and \
               len(self.Q_r_all_components[0]) == self.M_modal_Y:
                for q_factor_mode_m_plus_1 in self.Q_r_all_components[0]: # Iterate over Q_factors for the first component
                    output_Y_shape_dims.append(q_factor_mode_m_plus_1.shape[0]) # J_{m+1}
            elif self.M_modal_Y == 0: # Y was a vector (e.g. (I1,)), Y_pred should be (N_samples, 1)
                output_Y_shape_dims.append(1)
            else: # Fallback if Q_r factors are not available or M_modal_Y doesn't match
                  # This indicates an issue or an un-fitted model for tensor Y.
                  # Default to 1 for each of the M_modal_Y dimensions.
                for _ in range(self.M_modal_Y if self.M_modal_Y > 0 else 1): # ensure at least one trailing dim if M_modal_Y=0 initially led to problem
                    output_Y_shape_dims.append(1)
        
        # Ensure Y_pred is at least 2D (N_samples x 1)
        if len(output_Y_shape_dims) == 1: 
            output_Y_shape_dims.append(1)
        return torch.Size(output_Y_shape_dims)

    def predict(
        self, X_new: torch.Tensor, Y_true_for_shape_and_metric: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, int, List[float]]:
        
        current_device = X_new.device
        current_dtype = X_new.dtype

        output_Y_shape = self._determine_prediction_shape(X_new, Y_true_for_shape_and_metric)

        if self.T_mat is None or self.W_mat is None or self.num_components_fitted == 0:
            # Model not fitted or no components were fitted
            return torch.zeros(output_Y_shape, device=current_device, dtype=current_dtype), 0, []

        X_new_input_mat_form = X_new
        if X_new.ndim == 1: # Treat (N_samples,) as (N_samples, 1)
            X_new_input_mat_form = X_new.unsqueeze(1)
        
        X_new_mat = matricize_n(X_new_input_mat_form, mode=0) # Shape (N_new, prod(I2...IN))
        
        if self.W_mat.shape[0] != X_new_mat.shape[1]:
             raise ValueError(
                f"Feature dimension mismatch for prediction: "
                f"W_mat expects {self.W_mat.shape[0]} features (X modes other than samples), "
                f"but X_new provides {X_new_mat.shape[1]} features after unfolding."
            )

        T_new_all_r_mat = X_new_mat @ self.W_mat # Shape (N_new, R_fitted)

        best_q2_val = -float('inf')
        best_Y_pred_val: Optional[torch.Tensor] = None
        best_r_val = 0
        q2s_list: List[float] = []

        Y_true_dev: Optional[torch.Tensor] = None
        if Y_true_for_shape_and_metric is not None:
            Y_true_dev = Y_true_for_shape_and_metric
            if Y_true_dev.ndim == 1: Y_true_dev = Y_true_dev.unsqueeze(1) # Ensure at least 2D

        for r_to_use in range(1, self.num_components_fitted + 1):
            T_new_current_r_iter = T_new_all_r_mat[:, :r_to_use] # Shape (N_new, r_to_use)

            if self._is_matrix_Y:
                Q_r_subset = self.Q_r_all_components[:, :r_to_use] # Shape (M_Y, r_to_use)
                D_r_subset_coeffs = self.D_r_all_components[:r_to_use] # Shape (r_to_use,)
                # Y_pred = T_new @ diag(D_r) @ Q_r^T
                Y_pred_current_iter = (T_new_current_r_iter * D_r_subset_coeffs) @ Q_r_subset.T
            else: # Tensor Y
                Y_pred_current_iter = torch.zeros(output_Y_shape, device=current_device, dtype=current_dtype)
                for i_comp_idx in range(r_to_use):
                    t_new_for_comp = T_new_current_r_iter[:, i_comp_idx:i_comp_idx+1] # (N_new, 1)
                    D_comp = self.D_r_all_components[i_comp_idx] # Core (1, K2..KM)
                    Q_comp_list = self.Q_r_all_components[i_comp_idx] # List of Q factors for this component      
                    
                    factors_for_Y_approx = [t_new_for_comp] + Q_comp_list
                    term_Y_pred = tucker_to_tensor((D_comp, factors_for_Y_approx))
                    Y_pred_current_iter = Y_pred_current_iter + term_Y_pred
            
            if Y_true_dev is not None:
                current_q2_val = self.metric(Y_true_dev, Y_pred_current_iter)
                q2s_list.append(current_q2_val)
                if current_q2_val > best_q2_val:
                    best_q2_val = current_q2_val
                    best_Y_pred_val = Y_pred_current_iter
                    best_r_val = r_to_use
            elif r_to_use == self.num_components_fitted: # If no Y_true, take prediction from all fitted components
                best_Y_pred_val = Y_pred_current_iter 
                best_r_val = r_to_use
        
        if best_Y_pred_val is None: # Should only happen if num_components_fitted=0 (handled earlier) or Y_true_dev is None and loop didn't run
            if self.num_components_fitted > 0 and 'Y_pred_current_iter' in locals() and Y_pred_current_iter is not None:
                 best_Y_pred_val = Y_pred_current_iter
                 best_r_val = self.num_components_fitted
            else: # Fallback to zeros if no prediction could be made
                 best_Y_pred_val = torch.zeros(output_Y_shape, device=current_device, dtype=current_dtype)
                 best_r_val = 0
        
        return best_Y_pred_val, best_r_val, q2s_list

    def score(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        if self.T_mat is None: # Model not fitted
             # Optionally, fit here, or raise error. For now, let's assume predict handles it.
             # self.fit(X,Y) # This would modify the model state during score, which might be unexpected.
             pass
        
        _Y_pred, _best_r, q2s = self.predict(X, Y_true_for_shape_and_metric=Y)
        if not q2s: # No components or Y_true not provided to predict for metric calculation
            if _Y_pred is not None: # if predict returned a Y_pred even without Y_true (e.g. last component)
                return self.metric(Y, _Y_pred)
            return -float('inf') 
        return max(q2s)