import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from typing import Callable, Optional, Sequence, Tuple, Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed    
from sklearn.linear_model import Ridge
from .hopls_new import HOPLS as HOPLS_NEW  
import torch

def hopls_predictor(X_tr, y_tr, X_te, R=120, Ln=(8,8), epsilon=1e-9, print_shapes=False):
    """
    Tensor‐mode HOPLS predictor for a single rolling window.
    """
    import torch  # Ensure torch is available in this function's scope
    from .hopls_new import HOPLS as HOPLS_NEW
    model = HOPLS_NEW(R=R, Ln=list(Ln), epsilon=epsilon)
    model.fit(torch.Tensor(X_tr), torch.Tensor(y_tr))
    
    try:
        Y_pred, _, _ = model.predict(
        torch.Tensor(X_te),
        torch.Tensor(y_tr[: X_te.shape[0]])
    )
    except:
        X_te_tensor = torch.Tensor(X_te)
        num_test, n_series = X_te_tensor.shape[0], X_te_tensor.shape[1]
        y_init = torch.zeros((num_test, n_series), dtype=X_te_tensor.dtype)
        Y_pred, _, _ = model.predict(
            X_te_tensor,
            y_init
        )
    return Y_pred.detach().cpu().numpy()

def hopls_ridge_predictor(X_tr, y_tr, X_te, R=120, Ln=(8,8), epsilon=1e-9, alpha=1.0, print_shapes=False):
    """
    HOPLS + Ridge predictor for a single rolling window.
    """
    from .hopls_new import HOPLS_RIDGE
    model = HOPLS_RIDGE(R=R, Ln=list(Ln), epsilon=epsilon, ridge=alpha)
    model.fit(torch.Tensor(X_tr), torch.Tensor(y_tr))

    try:
        Y_pred, _, _ = model.predict(
        torch.Tensor(X_te),
        torch.Tensor(y_tr[: X_te.shape[0]])
    )
    except:
        X_te_tensor = torch.Tensor(X_te)
        num_test, n_series = X_te_tensor.shape[0], X_te_tensor.shape[1]
        y_init = torch.zeros((num_test, n_series), dtype=X_te_tensor.dtype)
        Y_pred, _, _ = model.predict(
            X_te_tensor,
            y_init
        )
    return Y_pred.detach().cpu().numpy()

def hopls_milr_predictor(X_tr, y_tr, X_te, R=120, Ln=(8,8), epsilon=1e-9, lambda_X=1e-3, lambda_Y=1e-3, alpha=1.0, print_shapes=False):
    """
    HOPLS + MILR predictor for a single rolling window.
    """
    from .hopls_new_new import HOPLS_MILR
    model = HOPLS_MILR(R=R, Ln=list(Ln), epsilon=epsilon, lambda_X=lambda_X, lambda_Y=lambda_Y, alpha=alpha)
    model.fit(torch.Tensor(X_tr), torch.Tensor(y_tr))

    try:
        Y_pred, _, _ = model.predict(
        torch.Tensor(X_te),
        torch.Tensor(y_tr[: X_te.shape[0]])
    )
    except:
        X_te_tensor = torch.Tensor(X_te)
        num_test, n_series = X_te_tensor.shape[0], X_te_tensor.shape[1]
        y_init = torch.zeros((num_test, n_series), dtype=X_te_tensor.dtype)
        Y_pred, _, _ = model.predict(
            X_te_tensor,
            y_init
        )
    return Y_pred.detach().cpu().numpy()

def ridge_predictor(X_tr, y_tr, X_te, alpha=1.0):
    """
    Matrix‐mode Ridge predictor for a single rolling window.
    """
    model = Ridge(alpha=alpha)
    model.fit(X_tr, y_tr)
    return model.predict(X_te)

def linear_regression_predictor(X_tr, y_tr, X_te):
    """
    Matrix‐mode Linear Regression predictor for a single rolling window.
    """
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    return model.predict(X_te)

def lightgbm_predictor(X_tr, y_tr, X_te, **kwargs):
    """
    Matrix-mode LightGBM predictor for a single rolling window.
    """
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        raise ImportError("lightgbm is not installed. Please install it to use this predictor.")
    model = LGBMRegressor(n_jobs=1, **kwargs)
    model.fit(X_tr, y_tr)
    return model.predict(X_te)

def get_final_data(crsp_location='./data/crsp_q_ccm_1.csv', 
                   wrdsapps_location='./data/wrdsapps_finratio.csv', 
                   output_location='./data/final_data.csv',
                   save_to_csv=True):
    """
    This function processes financial datasets to create a final dataset with specific features.
    
    Parameters:
    - crsp_location: Path to the CRSP dataset CSV file.
    - wrdsapps_location: Path to the WRDS Apps financial ratios dataset CSV file.
    - output_location: Path where the final processed dataset will be saved as a CSV file.
    
    Returns:
    - None
    """
    
    # Import datasets
    crsp_q_ccm = pd.read_csv(crsp_location)
    wrdsapps_finratio = pd.read_csv(wrdsapps_location)

    # Convert date columns to datetime
    crsp_q_ccm['datadate'] = pd.to_datetime(crsp_q_ccm['datadate'], errors='coerce')
    wrdsapps_finratio['public_date'] = pd.to_datetime(wrdsapps_finratio['public_date'], errors='coerce')

    # Convert to year-month format
    crsp_q_ccm['year_month'] = crsp_q_ccm['datadate'].dt.to_period('M')
    wrdsapps_finratio['year_month'] = wrdsapps_finratio['public_date'].dt.to_period('M')

    # Perform the inner join
    full_data = crsp_q_ccm.merge(
        wrdsapps_finratio,
        left_on=['GVKEY', 'year_month'],
        right_on=['gvkey', 'year_month'],
        how='inner'
    )

    # Drop redundant columns
    full_data.drop(columns=['gvkey', 'datadate', 'public_date'], inplace=True, errors='ignore')

    # Reorder columns
    cols = ['GVKEY', 'year_month'] + [col for col in full_data.columns if col not in ['GVKEY', 'year_month']]
    full_data = full_data[cols]

    # Sort by GVKEY and year_month
    full_data.sort_values(by=['GVKEY', 'year_month'], inplace=True)

    # Find GVKEYs with the most entries
    gvkey_counts = full_data.groupby("GVKEY")["year_month"].nunique()
    max_entries = gvkey_counts.max()
    gvkeys_with_max_entries = gvkey_counts[gvkey_counts == max_entries].index.tolist()

    # Filter data to keep only GVKEYs with max entries
    data_all_dates_trimmed = full_data[full_data["GVKEY"].isin(gvkeys_with_max_entries)]
    data_all_dates_trimmed.reset_index(drop=True, inplace=True)

    # Define the features we want to keep
    features = [
        'trt1m',  # Target variable

        # 1. Market-Related Factors (Macroeconomic & Market-wide)
        'divyield',  # Dividend Yield
        'bm',  # Book-to-Market Ratio
        'pe_exi', 'pe_inc',  # Price-to-Earnings Ratios
        'evm',  # Enterprise Value Multiple
        'de_ratio', 'debt_capital',  # Debt/Market Cap Ratios
        'ps',  # Price-to-Sales
        'ptb',  # Price-to-Book

        # 2. Profitability & Growth Factors
        'roe', 'roa', 'roce',  # Return on Equity, Assets, Capital Employed
        'gpm', 'npm', 'opmad', 'opmbd',  # Profit Margins (Gross, Net, Operating)
        'rd_sale',  # R&D to Sales
        'adv_sale',  # Advertising Expense to Sales
        'staff_sale',  # Labour Expense to Sales

        # 3. Risk & Leverage Factors
        'dltt_be',  # Long-term Debt/Book Equity 
        'debt_assets',  # Total Debt/Total Assets
        'debt_ebitda',  # Debt/EBITDA
        'intcov', 'intcov_ratio',  # Interest Coverage Ratios
        'ocf_lct',  # Operating CF/Current Liabilities
        'cash_debt',  # Cash Flow/Total Debt

        # 4. Liquidity & Efficiency Factors
        'at_turn',  # Asset Turnover
        'inv_turn',  # Inventory Turnover
        'rect_turn',  # Receivables Turnover
        'pay_turn',  # Payables Turnover
        'curr_ratio', 'quick_ratio', 'cash_ratio',  # Liquidity Ratios

        # 5. Size & Trading Activity
        'cshoq', 'cshom',  # Common Shares Outstanding
        'prccm',  # Market Price per Share (used for Market Cap calculation)
        'cshtrm',  # Trading Volume
        
        # 6. Sector Info
        'gsector' # GICS Sector code
    ]

    # Keep only the desired columns
    data_all_dates_trimmed = data_all_dates_trimmed[['GVKEY', 'year_month'] + features]

    # Identify companies with complete data (excluding cshoq)
    cols_to_check = [col for col in data_all_dates_trimmed.columns if col not in ['GVKEY', 'year_month', 'cshoq']]
    complete_gvkeys = data_all_dates_trimmed.groupby('GVKEY', group_keys=False).filter(
        lambda group: (group[cols_to_check].isna().mean() <= 0).all()
    )['GVKEY'].unique()


    # Create final_data with only companies that have complete data
    final_data = data_all_dates_trimmed[data_all_dates_trimmed['GVKEY'].isin(complete_gvkeys)].copy()
    final_data.reset_index(drop=True, inplace=True)

    # Ensure data is sorted by date within each company
    final_data.sort_values(['GVKEY', 'year_month'], inplace=True)

    # Interpolate missing cshoq values within each company group
    final_data['cshoq'] = final_data.groupby('GVKEY')['cshoq'].transform(lambda x: x.interpolate(method='linear'))

    # Fill remaining NaNs using backward and forward fill
    final_data['cshoq'] = final_data.groupby('GVKEY')['cshoq'].transform(lambda x: x.bfill().ffill())

    # Convert year_month from Period to string for CSV output
    final_data['year_month'] = final_data['year_month'].astype(str)

    # Save to CSV
    if save_to_csv:
        output_file = output_location
        final_data.to_csv(output_file, index=False)
    return final_data


class PredictionTestEngine:
    """
    A rolling-window test engine that can run multiple predictors in parallel.
    """

    def __init__(
        self,
        X_all: np.ndarray,
        y_all: np.ndarray,
        window_size: int,
        train_start: Optional[int] = None,
        time_index: Optional[Sequence] = None
    ):
        if X_all.ndim != 3:
            raise ValueError(f"X_all must be 3D (T, n_series, n_features), got {X_all.shape}")
        if y_all.ndim != 2:
            raise ValueError(f"y_all must be 2D (T, n_series), got {y_all.shape}")
        self.X_all = X_all
        self.y_all = y_all
        self.T, self.n_series, self.n_features = X_all.shape
        self.window_size = window_size
        self.train_start = train_start if train_start is not None else window_size
        if self.train_start < window_size:
            raise ValueError("train_start must be >= window_size to have a full initial window")
        self.time_index = time_index
        # precompute test indices once
        self.test_indices = list(range(self.train_start, self.T))

    def run_window(
        self,
        method: str = "hopls",
        verbose: bool = False,
        n_jobs: int = 1,
        **method_kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Sequence], Dict[str, float]]:
        """
        Run rolling-window predictions using the specified method.

        Parameters
        ----------
        method : str
            "hopls", "ridge", "hopls_ridge", or "linear_regression"
        verbose : bool
            Show a tqdm bar
        n_jobs : int
            Number of parallel processes (use 1 for serial)
        **method_kwargs :
            All other kwargs are forwarded to the predictor (e.g. R, Ln, epsilon or alpha)
        """
        if verbose:
            print(f"run_window: X_all shape {self.X_all.shape}, y_all shape {self.y_all.shape}")
            print(f"run_window: window_size={self.window_size}, n_series={self.n_series}, n_features={self.n_features}")
            print(f"run_window: number of test windows={len(self.test_indices)}")
        # map names to functions & modes
        predictor_map = {
            "hopls": hopls_predictor,
            "ridge": ridge_predictor,
            "hopls_ridge": hopls_ridge_predictor,
            "linear_regression": linear_regression_predictor,
            "hopls_milr": hopls_milr_predictor
        }
        mode_map = {
            "hopls": "tensor",
            "ridge": "matrix",
            "hopls_ridge": "tensor",
            "linear_regression": "matrix",
            "hopls_milr": "tensor"
        }
        # add LightGBM mapping
        predictor_map["lightgbm"] = lightgbm_predictor
        # add LightGBM mode
        mode_map["lightgbm"] = "matrix"

        if method not in predictor_map:
            raise ValueError(f"Unknown method '{method}', choose 'hopls', 'ridge', 'hopls_ridge', or 'linear_regression'")
        predictor = predictor_map[method]
        mode = mode_map[method]

        preds = [None] * len(self.test_indices)
        indices = list(enumerate(self.test_indices))

        # force single worker for LightGBM
        if method == "lightgbm":
            n_jobs = 1

        if n_jobs == 1:
            loop = tqdm(indices, desc="Rolling prediction") if verbose else indices
            for i, t in loop:
                preds[i] = self._predict_single(t, predictor, mode, method_kwargs)
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as exe:
                futures = {
                    exe.submit(self._predict_single, t, predictor, mode, method_kwargs): i
                    for i, t in indices
                }
                iterator = (
                    tqdm(as_completed(futures), total=len(futures), desc="Parallel preds")
                    if verbose else as_completed(futures)
                )
                for fut in iterator:
                    i = futures[fut]
                    preds[i] = fut.result()

        y_pred_all = np.stack(preds, axis=0)
        y_true_all = self.y_all[self.test_indices]
        time_index_test = (
            self.time_index[self.train_start:]
            if self.time_index is not None else None
        )

        # compute metrics
        mse = mean_squared_error(y_true_all.ravel(), y_pred_all.ravel())
        r2 = r2_score(y_true_all.ravel(), y_pred_all.ravel())
        true_dir = np.sign(y_true_all.ravel())
        pred_dir = np.sign(y_pred_all.ravel())
        mask = true_dir != 0
        directional_acc = (
            np.mean(pred_dir[mask] == true_dir[mask])
            if mask.any() else np.nan
        )
        metrics = {
            "mse": mse,
            "r2": r2,
            "directional_accuracy": directional_acc
        }

        # store for plotting
        self.y_pred_all = y_pred_all
        self.y_true_all = y_true_all
        self.time_index_test = time_index_test
        self.metrics = metrics

        # if verbose tensor-mode, print HOPLS shapes for last window only
        if verbose and mode == 'tensor':
            # prepare last window data for shape inspection
            last_t = self.test_indices[-1]
            start = last_t - self.window_size
            X_win = self.X_all[start:last_t]
            y_win = self.y_all[start:last_t]
            # instantiate and fit model once to inspect core tensor shapes
            if method == 'hopls_milr':
                from .hopls_new_new import HOPLS_MILR
                model_ins = HOPLS_MILR(**{k: method_kwargs[k] for k in ['R', 'Ln', 'epsilon', 'lambda_X', 'lambda_Y', 'alpha']})
            else:
                from .hopls_new import HOPLS as HOPLS_INS
                model_ins = HOPLS_INS(**{k: method_kwargs[k] for k in ['R', 'Ln', 'epsilon']})
            import torch
            model_ins.fit(torch.Tensor(X_win), torch.Tensor(y_win))
            _print_hopls_shapes(model_ins)

        return y_pred_all, y_true_all, time_index_test, metrics

    def run_split(
        self,
        train_split: float = 0.8,
        method: str = "hopls",
        verbose: bool = False,
        **method_kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Sequence], Dict[str, float]]:
        """
        Run predictions using a single train/test split instead of rolling windows.

        Parameters
        ----------
        train_split : float
            Proportion of data to use for training (e.g., 0.8 for 80% train, 20% test)
        method : str
            "hopls", "ridge", "hopls_ridge", "linear_regression", or "hopls_milr"
        verbose : bool
            Show progress information
        **method_kwargs :
            All other kwargs are forwarded to the predictor (e.g. R, Ln, epsilon or alpha)

        Returns
        -------
        y_pred : np.ndarray
            Predicted y values for test set, shape (n_test, n_series).
        y_true : np.ndarray
            True y values for test set, shape (n_test, n_series).
        time_index_test : Sequence or None
            Timestamps for test set.
        metrics : Dict[str, float]
            MSE, R2, and directional accuracy.
        """
        # map names to functions & modes
        predictor_map = {
            "hopls": hopls_predictor,
            "ridge": ridge_predictor,
            "hopls_ridge": hopls_ridge_predictor,
            "linear_regression": linear_regression_predictor,
            "hopls_milr": hopls_milr_predictor,
            "lightgbm": lightgbm_predictor
        }
        mode_map = {
            "hopls": "tensor",
            "ridge": "matrix",
            "hopls_ridge": "tensor",
            "linear_regression": "matrix",
            "hopls_milr": "tensor",
            "lightgbm": "matrix"
        }

        if method not in predictor_map:
            raise ValueError(f"Unknown method '{method}', available: {list(predictor_map.keys())}")
        
        predictor = predictor_map[method]
        mode = mode_map[method]

        # Calculate split point
        split_idx = int(train_split * self.T)
        if split_idx <= 0 or split_idx >= self.T:
            raise ValueError(f"Invalid train_split {train_split}, results in split_idx={split_idx} for T={self.T}")

        if verbose:
            print(f"Using train/test split: {split_idx}/{self.T - split_idx} (train/test)")

        # Split data
        X_train = self.X_all[:split_idx]  # shape (train_size, n_series, n_features)
        y_train = self.y_all[:split_idx]  # shape (train_size, n_series)
        X_test = self.X_all[split_idx:]   # shape (test_size, n_series, n_features)
        y_true = self.y_all[split_idx:]   # shape (test_size, n_series)

        if mode == "matrix":
            # flatten training data: (train_size * n_series) × n_features
            X_tr = X_train.reshape(-1, self.n_features)
            y_tr = y_train.reshape(-1)
            # flatten test data: (test_size * n_series) × n_features
            X_te = X_test.reshape(-1, self.n_features)
            
            if verbose:
                print(f"Matrix mode: X_train shape {X_tr.shape}, y_train shape {y_tr.shape}")
                print(f"Matrix mode: X_test shape {X_te.shape}")
            
            y_pred_flat = predictor(X_tr, y_tr, X_te, **method_kwargs)
            # reshape back to (test_size, n_series)
            y_pred = y_pred_flat.reshape(X_test.shape[0], X_test.shape[1])

        else:  # tensor mode
            if verbose:
                print(f"Tensor mode: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
                print(f"Tensor mode: X_test shape {X_test.shape}")
            
            y_pred = predictor(X_train, y_train, X_test, **method_kwargs)
            y_pred = np.asarray(y_pred)

        # Time index for test set
        time_index_test = (
            self.time_index[split_idx:]
            if self.time_index is not None else None
        )

        # Compute metrics
        mse = mean_squared_error(y_true.ravel(), y_pred.ravel())
        r2 = r2_score(y_true.ravel(), y_pred.ravel())
        true_dir = np.sign(y_true.ravel())
        pred_dir = np.sign(y_pred.ravel())
        mask = true_dir != 0
        directional_acc = (
            np.mean(pred_dir[mask] == true_dir[mask])
            if mask.any() else np.nan
        )
        metrics = {
            "mse": mse,
            "r2": r2,
            "directional_accuracy": directional_acc
        }

        # Store for plotting
        self.y_pred_all = y_pred
        self.y_true_all = y_true
        self.time_index_test = time_index_test
        self.metrics = metrics

        if verbose:
            print(f"Split results - MSE: {mse:.6f}, R2: {r2:.6f}, Dir. Acc.: {directional_acc:.6f}")

        return y_pred, y_true, time_index_test, metrics

    def _predict_single(
        self,
        t: int,
        predictor: Callable,
        mode: str,
        kw: Dict[str, Any]
    ) -> np.ndarray:
        """
        Slice out the window at time t and call the predictor.
        """
        start = t - self.window_size
        X_win = self.X_all[start:t]     # shape (window_size, n_series, n_features)
        y_win = self.y_all[start:t]     # shape (window_size, n_series)

        if mode == "matrix":
            # flatten into (window_size*n_series) × n_features for training
            X_tr = X_win.reshape(-1, self.n_features)
            y_tr = y_win.reshape(-1)
            # test = current slice shaped series × features
            X_te = self.X_all[t].reshape(self.n_series, self.n_features)
            return predictor(X_tr, y_tr, X_te, **kw)

        else:  # tensor mode
            X_tr = X_win
            y_tr = y_win
            X_te = self.X_all[t][None, ...]  # add leading dim → (1, series, features)
            y_pred = predictor(X_tr, y_tr, X_te, **kw)
            return np.asarray(y_pred).squeeze(0)


    def plot_results(self, series_indices: Optional[Sequence[int]] = None):
        """
        Plot true vs predicted for selected series, two ways:
          1. Predicted shifted one step ahead (convention)
          2. Predicted aligned back to true values (for magnitude comparison)
        """
        if not hasattr(self, 'y_pred_all'):
            raise RuntimeError("No predictions found. Run run() first.")
        y_pred = self.y_pred_all
        y_true = self.y_true_all
        times = np.asarray(self.time_index_test) if self.time_index_test is not None else np.arange(len(y_true))
        if series_indices is None:
            series_indices = list(range(min(3, self.n_series)))

        # 1. shifted one step ahead
        plt.figure(figsize=(12, 6))
        for idx in series_indices:
            plt.plot(times, y_true[:, idx], label=f"Actual series {idx}")
            plt.plot(times, y_pred[:, idx], '--', label=f"Predicted (shifted) series {idx}")
        plt.xlabel("Time")
        plt.ylabel("Target value")
        plt.title("Predicted vs Actual (shifted one step ahead)")
        plt.legend()
        plt.show()

        # 2. aligned back to true
        aligned_pred = np.roll(y_pred, -1, axis=0)
        aligned_times = times[:-1]
        plt.figure(figsize=(12, 6))
        for idx in series_indices:
            plt.plot(aligned_times, y_true[:-1, idx], label=f"Actual series {idx}")
            plt.plot(aligned_times, aligned_pred[:-1, idx], '--', label=f"Predicted aligned series {idx}")
        plt.xlabel("Time")
        plt.ylabel("Target value")
        plt.title("Predicted vs Actual (aligned back)")
        plt.legend()
        plt.show()

    def summary(self) -> Dict[str, Any]:
        """Return summary of test metrics."""
        if not hasattr(self, 'metrics'):
            raise RuntimeError("No metrics found. Run run() first.")
        return self.metrics

    def run_comparison(
        self,
        methods_params: Dict[str, Dict[str, Any]],
        n_jobs: int = 1,
        verbose: bool = False,
        series_indices: Optional[Sequence[int]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run and compare multiple prediction methods with given parameters, print metrics,
        and plot actual vs predicted for specified series on subplots.
        """
        import matplotlib.pyplot as plt
        results: Dict[str, Dict[str, Any]] = {}
        for method, params in methods_params.items():
            y_pred, y_true, times, metrics = self.run_window(
                method=method,
                verbose=verbose,
                n_jobs=n_jobs,
                **params
            )
            results[method] = {
                'y_pred': y_pred,
                'y_true': y_true,
                'times': times,
                'metrics': metrics
            }
            print(f"Metrics for {method}: {metrics}")
        # default series indices to first two if not provided
        if series_indices is None:
            series_indices = [0, 1]
        # plot comparisons in one figure per method
        fig, axes = plt.subplots(
            nrows=len(results),
            ncols=1,
            figsize=(12, 6 * len(results)),
            sharex=True
        )
        if len(results) == 1:
            axes = [axes]
        for ax, (method, res) in zip(axes, results.items()):
            y_pred = res['y_pred']
            y_true = res['y_true']
            times = res['times']
            for idx in series_indices:
                ax.plot(times, y_true[:, idx], label=f"Actual series {idx}")
                ax.plot(times, y_pred[:, idx], '--', label=f"{method} predicted series {idx}")
            ax.set_title(f"Comparison for {method}")
            ax.legend()
        plt.tight_layout()
        plt.show()
        return results

def _print_hopls_shapes(model):
    """
    Print key tensor shapes for HOPLS and HOPLS-MILR models.
    """
    # HOPLS / HOPLS-RIDGE
    if hasattr(model, 'model'):
        P_list, Q_mat, D, T_mat, W_mat = model.model
        print(f"Components: {len(P_list)}")
        print("P shapes per component:")
        for i, Pr in enumerate(P_list):
            print(f"  Comp {i}: {[tuple(p.shape) for p in Pr]}")
        print(f"Q_mat shape: {tuple(Q_mat.shape)}, D shape: {tuple(D.shape)}")
        print(f"T_mat shape: {tuple(T_mat.shape)}, W_mat shape: {tuple(W_mat.shape)}")
        return
    # HOPLS-MILR
    if hasattr(model, 'P_r_all_components'):
        P_all = model.P_r_all_components
        G_all = model.G_r_all_components
        Qr = model.Q_r_all_components
        D = model.D_r_all_components
        T = model.T_mat
        W = model.W_mat
        print(f"MILR components: {len(P_all)}")
        for i, Pr in enumerate(P_all):
            print(f"  P comp {i}: {[tuple(p.shape) for p in Pr]}")
        print(f"G shapes: {[tuple(G.shape) for G in G_all]}")
        print(f"Q shape: {tuple(Qr.shape)}")
        print(f"D shape: {tuple(D.shape)}")
        print(f"T_mat shape: {tuple(T.shape)}, W_mat shape: {tuple(W.shape)}")
        return
    print("No HOPLS model attributes found to print shapes.")
