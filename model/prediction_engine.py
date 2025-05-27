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

def hopls_predictor(X_tr, y_tr, X_te, R=120, Ln=(8,8), epsilon=1e-9):
    """
    Tensor‐mode HOPLS predictor for a single rolling window.
    """
    import torch  # Ensure torch is available in this function's scope
    from .hopls_new import HOPLS as HOPLS_NEW
    model = HOPLS_NEW(R=R, Ln=list(Ln), epsilon=epsilon)
    model.fit(torch.Tensor(X_tr), torch.Tensor(y_tr))
    Y_pred, _, _ = model.predict(
        torch.Tensor(X_te),
        torch.Tensor(y_tr[: X_te.shape[0]])
    )
    return Y_pred.detach().cpu().numpy()

def hopls_ridge_predictor(X_tr, y_tr, X_te, R=120, Ln=(8,8), epsilon=1e-9, alpha=1.0):
    """
    HOPLS + Ridge predictor for a single rolling window.
    """
    import torch  # Ensure torch is available in this function's scope
    from .hopls_new import HOPLS_RIDGE
    model = HOPLS_RIDGE(R=R, Ln=list(Ln), epsilon=epsilon, ridge=alpha)
    model.fit(torch.Tensor(X_tr), torch.Tensor(y_tr))
    Y_pred, _, _ = model.predict(
        torch.Tensor(X_te),
        torch.Tensor(y_tr[: X_te.shape[0]])
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

    def run(
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
        # map names to functions & modes
        predictor_map = {
            "hopls": hopls_predictor,
            "ridge": ridge_predictor,
            "hopls_ridge": hopls_ridge_predictor,
            "linear_regression": linear_regression_predictor
        }
        mode_map = {
            "hopls": "tensor",
            "ridge": "matrix",
            "hopls_ridge": "tensor",
            "linear_regression": "matrix"
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

        return y_pred_all, y_true_all, time_index_test, metrics

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


# def run(
    #     self,
    #     predictor: Callable,
    #     verbose: bool = False,
    #     **predictor_kwargs: Any
    # ) -> Tuple[np.ndarray, np.ndarray, Optional[Sequence], Dict[str, float]]:
    #     """
    #     Run rolling-window predictions.

    #     predictor signature:
    #       - matrix mode: (X_train_mat, y_train_vec, X_test_mat, **kwargs) -> y_pred_vec
    #       - tensor mode: (X_train_tensor, y_train_tensor, X_test_tensor, **kwargs) -> y_pred_tensor

    #     Returns
    #     -------
    #     y_pred_all : np.ndarray
    #         Predicted y values, shape (n_tests, n_series).
    #     y_true_all : np.ndarray
    #         True y values, shape (n_tests, n_series).
    #     time_index_test : Sequence or None
    #         Timestamps for test set, length n_tests.
    #     metrics : Dict[str, float]
    #         Overall MSE, R2, and directional accuracy.
    #     """
    #     preds = []
    #     loop = self.test_indices
    #     if verbose:
    #         try:
    #             loop = tqdm(loop, desc="Rolling prediction")
    #         except ImportError:
    #             print("tqdm not installed, running without progress bar.")
    #     for t in loop:
    #         start = t - self.window_size
    #         end = t
    #         X_win = self.X_all[start:end]  # shape (window_size, series, features)
    #         y_win = self.y_all[start:end]  # shape (window_size, series)
    #         if self.mode == 'matrix':
    #             X_train_mat = X_win.reshape(-1, self.n_features)
    #             y_train_vec = y_win.reshape(-1)
    #             X_test_mat = self.X_all[t].reshape(self.n_series, self.n_features)
    #             y_pred = predictor(X_train_mat, y_train_vec, X_test_mat, **predictor_kwargs)
    #         elif self.mode == 'tensor':
    #             X_train_tensor = X_win
    #             y_train_tensor = y_win
    #             X_test_tensor = self.X_all[t][np.newaxis, ...]
    #             y_pred_tensor = predictor(X_train_tensor, y_train_tensor, X_test_tensor, **predictor_kwargs)
    #             y_pred = np.asarray(y_pred_tensor).squeeze(0)
    #         else:
    #             raise ValueError("mode must be 'matrix' or 'tensor'")
    #         preds.append(y_pred)

    #     y_pred_all = np.stack(preds, axis=0)
    #     y_true_all = self.y_all[self.test_indices]
    #     time_index_test = None
    #     if self.time_index is not None:
    #         time_index_test = self.time_index[self.train_start:]

    #     # compute metrics
    #     mse = mean_squared_error(y_true_all.reshape(-1), y_pred_all.reshape(-1))
    #     r2 = r2_score(y_true_all.reshape(-1), y_pred_all.reshape(-1))
    #     # directional accuracy: proportion where sign matches (ignore zeros in true)
    #     true_dir = np.sign(y_true_all.reshape(-1))
    #     pred_dir = np.sign(y_pred_all.reshape(-1))
    #     mask = true_dir != 0
    #     directional_acc = np.mean(pred_dir[mask] == true_dir[mask]) if np.any(mask) else np.nan
    #     metrics = {'mse': mse, 'r2': r2, 'directional_accuracy': directional_acc}

    #     # store for plotting
    #     self.y_pred_all = y_pred_all
    #     self.y_true_all = y_true_all
    #     self.time_index_test = time_index_test
    #     self.metrics = metrics

    #     return y_pred_all, y_true_all, time_index_test, metrics
