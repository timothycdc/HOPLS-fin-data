import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from typing import Callable, Optional, Sequence, Tuple, Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from sklearn.linear_model import Ridge
from .hopls import HOPLS as HOPLS_NEW
from ..archive.hopls_milr_rhooi import HOPLS_MILR_RHOOI
import torch


def hopls_predictor(
    X_tr, y_tr, X_te, R=120, Ln=(8, 8), epsilon=1e-9, print_shapes=False
):
    """
    Tensor‐mode HOPLS predictor for a single rolling window.
    """
    import torch  # Ensure torch is available in this function's scope
    from .hopls import HOPLS as HOPLS_NEW

    model = HOPLS_NEW(R=R, Ln=list(Ln), epsilon=epsilon)
    model.fit(torch.Tensor(X_tr), torch.Tensor(y_tr))

    try:
        Y_pred, _, _ = model.predict(
            torch.Tensor(X_te), torch.Tensor(y_tr[: X_te.shape[0]])
        )
    except:
        X_te_tensor = torch.Tensor(X_te)
        num_test, n_series = X_te_tensor.shape[0], X_te_tensor.shape[1]
        y_init = torch.zeros((num_test, n_series), dtype=X_te_tensor.dtype)
        Y_pred, _, _ = model.predict(X_te_tensor, y_init)
    return Y_pred.detach().cpu().numpy()


def hopls_ridge_predictor(
    X_tr, y_tr, X_te, R=120, Ln=(8, 8), epsilon=1e-9, lambda_X=1e-3, lambda_Y = 1e-3, print_shapes=False
):
    """
    HOPLS + Ridge predictor for a single rolling window.
    """
    from .hopls_ridge import HOPLS_RIDGE

    model = HOPLS_RIDGE(R=R, Ln=list(Ln), epsilon=epsilon, lambda_X=lambda_X, lambda_Y=lambda_Y)
    model.fit(torch.Tensor(X_tr), torch.Tensor(y_tr))

    try:
        Y_pred, _, _ = model.predict(
            torch.Tensor(X_te), torch.Tensor(y_tr[: X_te.shape[0]])
        )
    except:
        X_te_tensor = torch.Tensor(X_te)
        num_test, n_series = X_te_tensor.shape[0], X_te_tensor.shape[1]
        y_init = torch.zeros((num_test, n_series), dtype=X_te_tensor.dtype)
        Y_pred, _, _ = model.predict(X_te_tensor, y_init)
    return Y_pred.detach().cpu().numpy()


def hopls_milr_predictor(
    X_tr,
    y_tr,
    X_te,
    R=120,
    Ln=(8, 8),
    epsilon=1e-9,
    lambda_X=1e-3,
    lambda_Y=1e-3,
    alpha=1.0,
    print_shapes=False,
):
    """
    HOPLS + MILR predictor for a single rolling window.
    """
    from .hopls_milr import HOPLS_MILR

    model = HOPLS_MILR(
        R=R,
        Ln=list(Ln),
        epsilon=epsilon,
        lambda_X=lambda_X,
        lambda_Y=lambda_Y,
        alpha=alpha,
    )
    model.fit(torch.Tensor(X_tr), torch.Tensor(y_tr))

    try:
        Y_pred, _, _ = model.predict(
            torch.Tensor(X_te), torch.Tensor(y_tr[: X_te.shape[0]])
        )
    except:
        X_te_tensor = torch.Tensor(X_te)
        num_test, n_series = X_te_tensor.shape[0], X_te_tensor.shape[1]
        y_init = torch.zeros((num_test, n_series), dtype=X_te_tensor.dtype)
        Y_pred, _, _ = model.predict(X_te_tensor, y_init)
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
        raise ImportError(
            "lightgbm is not installed. Please install it to use this predictor."
        )
    model = LGBMRegressor(n_jobs=1, **kwargs)
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def hopls_milr_rhooi_predictor(
    X_tr,
    y_tr,
    X_te,
    R=60,
    Ln=(8, 8),
    Km=None,
    lambda_X=1e-3,
    lambda_Y=1e-3,
    alpha=1.0,
    lambda_P_factor_penalty=None,
    lambda_Q_factor_penalty=None,
    epsilon=1e-9,
    rhooi_n_iter_max=100,
    rhooi_tol=1e-7,
    rhooi_verbose=False,
    print_shapes=False,
):
    """
    HOPLS + MILR + RHOOI predictor for a single rolling window.
    Uses tensor mode for X and Y.
    """
    # HOPLS_MILR_RHOOI is imported at the top of the file

    model = HOPLS_MILR_RHOOI(
        R=R,
        Ln=list(Ln),
        Km=Km,  # Km=None is handled by the class for matrix Y
        lambda_X=lambda_X,
        lambda_Y=lambda_Y,
        alpha=alpha,
        lambda_P_factor_penalty=lambda_P_factor_penalty,
        lambda_Q_factor_penalty=lambda_Q_factor_penalty,
        epsilon=epsilon,
        rhooi_n_iter_max=rhooi_n_iter_max,
        rhooi_tol=rhooi_tol,
        # rhooi_orthogonality=rhooi_orthogonality,
        rhooi_verbose=rhooi_verbose,
    )

    model.fit(torch.Tensor(X_tr), torch.Tensor(y_tr))
    X_te_tensor = torch.Tensor(X_te)

    try:
        # Try predicting without providing Y_true_for_shape_and_metric
        # Assumes model.predict can handle this (e.g., by using all fitted components
        # or selecting the last component if Q2 cannot be computed)
        y_pred_tensor, _, _ = model.predict(
            X_te_tensor, Y_true_for_shape_and_metric=None
        )
    except Exception:
        # Fallback: if predict(..., None) fails, try with a dummy y_init.
        # This matches the pattern in other predictors.
        num_test_samples = X_te_tensor.shape[0]

        # For tensor mode, y_tr should be (train_samples, n_series_out)
        n_series_out = y_tr.shape[1] if y_tr.ndim > 1 else 1

        y_init_shape = (num_test_samples, n_series_out)
        # Ensure y_init is on the same device and dtype as model tensors
        y_init = torch.zeros(
            y_init_shape, dtype=torch.Tensor(
                X_tr).dtype, device=X_te_tensor.device
        )

        y_pred_tensor, _, _ = model.predict(
            X_te_tensor, Y_true_for_shape_and_metric=y_init
        )

    return y_pred_tensor.detach().cpu().numpy()


def get_final_data(
    crsp_location="./data/crsp_q_ccm_1.csv",
    wrdsapps_location="./data/wrdsapps_finratio.csv",
    output_location="./data/final_data.csv",
    save_to_csv=True,
):
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
    crsp_q_ccm["datadate"] = pd.to_datetime(
        crsp_q_ccm["datadate"], errors="coerce")
    wrdsapps_finratio["public_date"] = pd.to_datetime(
        wrdsapps_finratio["public_date"], errors="coerce"
    )

    # Convert to year-month format
    crsp_q_ccm["year_month"] = crsp_q_ccm["datadate"].dt.to_period("M")
    wrdsapps_finratio["year_month"] = wrdsapps_finratio["public_date"].dt.to_period(
        "M")

    # Perform the inner join
    full_data = crsp_q_ccm.merge(
        wrdsapps_finratio,
        left_on=["GVKEY", "year_month"],
        right_on=["gvkey", "year_month"],
        how="inner",
    )

    # Drop redundant columns
    full_data.drop(
        columns=["gvkey", "datadate", "public_date"], inplace=True, errors="ignore"
    )

    # Reorder columns
    cols = ["GVKEY", "year_month"] + [
        col for col in full_data.columns if col not in ["GVKEY", "year_month"]
    ]
    full_data = full_data[cols]

    # Sort by GVKEY and year_month
    full_data.sort_values(by=["GVKEY", "year_month"], inplace=True)

    # Find GVKEYs with the most entries
    gvkey_counts = full_data.groupby("GVKEY")["year_month"].nunique()
    max_entries = gvkey_counts.max()
    gvkeys_with_max_entries = gvkey_counts[gvkey_counts == max_entries].index.tolist(
    )

    # Filter data to keep only GVKEYs with max entries
    data_all_dates_trimmed = full_data[full_data["GVKEY"].isin(
        gvkeys_with_max_entries)]
    data_all_dates_trimmed.reset_index(drop=True, inplace=True)

    # Define the features we want to keep
    features = [
        "trt1m",  # Target variable
        # 1. Market-Related Factors (Macroeconomic & Market-wide)
        "divyield",  # Dividend Yield
        "bm",  # Book-to-Market Ratio
        "pe_exi",
        "pe_inc",  # Price-to-Earnings Ratios
        "evm",  # Enterprise Value Multiple
        "de_ratio",
        "debt_capital",  # Debt/Market Cap Ratios
        "ps",  # Price-to-Sales
        "ptb",  # Price-to-Book
        # 2. Profitability & Growth Factors
        "roe",
        "roa",
        "roce",  # Return on Equity, Assets, Capital Employed
        "gpm",
        "npm",
        "opmad",
        "opmbd",  # Profit Margins (Gross, Net, Operating)
        "rd_sale",  # R&D to Sales
        "adv_sale",  # Advertising Expense to Sales
        "staff_sale",  # Labour Expense to Sales
        # 3. Risk & Leverage Factors
        "dltt_be",  # Long-term Debt/Book Equity
        "debt_assets",  # Total Debt/Total Assets
        "debt_ebitda",  # Debt/EBITDA
        "intcov",
        "intcov_ratio",  # Interest Coverage Ratios
        "ocf_lct",  # Operating CF/Current Liabilities
        "cash_debt",  # Cash Flow/Total Debt
        # 4. Liquidity & Efficiency Factors
        "at_turn",  # Asset Turnover
        "inv_turn",  # Inventory Turnover
        "rect_turn",  # Receivables Turnover
        "pay_turn",  # Payables Turnover
        "curr_ratio",
        "quick_ratio",
        "cash_ratio",  # Liquidity Ratios
        # 5. Size & Trading Activity
        "cshoq",
        "cshom",  # Common Shares Outstanding
        "prccm",  # Market Price per Share (used for Market Cap calculation)
        "cshtrm",  # Trading Volume
        # 6. Sector Info
        "gsector",  # GICS Sector code
    ]

    # Keep only the desired columns
    data_all_dates_trimmed = data_all_dates_trimmed[[
        "GVKEY", "year_month"] + features]

    # Identify companies with complete data (excluding cshoq)
    cols_to_check = [
        col
        for col in data_all_dates_trimmed.columns
        if col not in ["GVKEY", "year_month", "cshoq"]
    ]
    complete_gvkeys = (
        data_all_dates_trimmed.groupby("GVKEY", group_keys=False)
        .filter(lambda group: (group[cols_to_check].isna().mean() <= 0).all())["GVKEY"]
        .unique()
    )

    # Create final_data with only companies that have complete data
    final_data = data_all_dates_trimmed[
        data_all_dates_trimmed["GVKEY"].isin(complete_gvkeys)
    ].copy()
    final_data.reset_index(drop=True, inplace=True)

    # Ensure data is sorted by date within each company
    final_data.sort_values(["GVKEY", "year_month"], inplace=True)

    # Interpolate missing cshoq values within each company group
    final_data["cshoq"] = final_data.groupby("GVKEY")["cshoq"].transform(
        lambda x: x.interpolate(method="linear")
    )

    # Fill remaining NaNs using backward and forward fill
    final_data["cshoq"] = final_data.groupby("GVKEY")["cshoq"].transform(
        lambda x: x.bfill().ffill()
    )

    # Convert year_month from Period to string for CSV output
    final_data["year_month"] = final_data["year_month"].astype(str)

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
        train_end: Optional[int] = None,
        time_index: Optional[Sequence] = None,
    ):
        if X_all.ndim != 3:
            raise ValueError(
                f"X_all must be 3D (T, n_series, n_features), got {X_all.shape}"
            )
        if y_all.ndim != 2:
            raise ValueError(
                f"y_all must be 2D (T, n_series), got {y_all.shape}")
        self.X_all = X_all
        self.y_all = y_all
        self.T, self.n_series, self.n_features = X_all.shape
        self.window_size = window_size
        self.train_start = train_start if train_start is not None else window_size
        # set train_end boundary for rolling windows
        self.train_end = train_end if train_end is not None else self.T
        if self.train_start < window_size:
            raise ValueError(
                "train_start must be >= window_size to have a full initial window"
            )
        self.time_index = time_index
        # precompute test indices once, now up to train_end
        self.test_indices = list(range(self.train_start, min(self.train_end, self.T)))

    def run_window(
        self,
        method: str = "hopls",
        verbose: bool = False,
        n_jobs: int = 1,
        **method_kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Sequence], Dict[str, float]]:
        """
        Run predictions using rolling windows.

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
            print(
                f"run_window: X_all shape {self.X_all.shape}, y_all shape {self.y_all.shape}"
            )
            print(
                f"run_window: window_size={self.window_size}, n_series={self.n_series}, n_features={self.n_features}"
            )
            print(
                f"run_window: number of test windows={len(self.test_indices)}")
            # Print test period index and date range if available
            if self.time_index is not None and self.test_indices:
                start_idx = self.test_indices[0]
                end_idx = self.test_indices[-1]
                start_date = self.time_index[start_idx]
                end_date = self.time_index[end_idx]
                print(f"run_window: test indices {start_idx} to {end_idx}, dates {start_date} to {end_date}")
        # map names to functions & modes
        predictor_map = {
            "hopls": hopls_predictor,
            "ridge": ridge_predictor,
            "hopls_ridge": hopls_ridge_predictor,
            "linear_regression": linear_regression_predictor,
            "hopls_milr": hopls_milr_predictor,
            "lightgbm": lightgbm_predictor,
            "hopls_milr_rhooi": hopls_milr_rhooi_predictor,
        }
        mode_map = {
            "hopls": "tensor",
            "ridge": "matrix",
            "hopls_ridge": "tensor",
            "linear_regression": "matrix",
            "hopls_milr": "tensor",
            "lightgbm": "matrix",
            "hopls_milr_rhooi": "tensor",
        }

        if method not in predictor_map:
            raise ValueError(
                f"Unknown method '{method}', choose 'hopls', 'ridge', 'hopls_ridge', or 'linear_regression'"
            )
        predictor = predictor_map[method]
        mode = mode_map[method]

        preds = [None] * len(self.test_indices)
        indices = list(enumerate(self.test_indices))

        # force single worker for LightGBM
        if method == "lightgbm":
            n_jobs = 1

        if n_jobs == 1:
            loop = tqdm(
                indices, desc="Rolling prediction") if verbose else indices
            for i, t in loop:
                preds[i] = self._predict_single(
                    t, predictor, mode, method_kwargs)
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as exe:
                futures = {
                    exe.submit(
                        self._predict_single, t, predictor, mode, method_kwargs
                    ): i
                    for i, t in indices
                }
                iterator = (
                    tqdm(
                        as_completed(futures), total=len(futures), desc="Parallel preds"
                    )
                    if verbose
                    else as_completed(futures)
                )
                for fut in iterator:
                    i = futures[fut]
                    preds[i] = fut.result()

        y_pred_all = np.stack(preds, axis=0)
        y_true_all = self.y_all[self.test_indices]
        time_index_test = (
            [self.time_index[i] for i in self.test_indices] if self.time_index is not None else None
        )

        # compute metrics
        mse = mean_squared_error(y_true_all.ravel(), y_pred_all.ravel())
        r2 = r2_score(y_true_all.ravel(), y_pred_all.ravel())
        true_dir = np.sign(y_true_all.ravel())
        pred_dir = np.sign(y_pred_all.ravel())
        mask = true_dir != 0
        directional_acc = (
            np.mean(pred_dir[mask] == true_dir[mask]) if mask.any() else np.nan
        )
        metrics = {"mse": mse, "r2": r2,
                   "directional_accuracy": directional_acc}

        # store for plotting
        self.y_pred_all = y_pred_all
        self.y_true_all = y_true_all
        self.time_index_test = time_index_test
        self.metrics = metrics

        # if verbose tensor-mode, print HOPLS shapes for last window only
        if verbose and mode == "tensor":
            # prepare last window data for shape inspection
            last_t = self.test_indices[-1]
            start = last_t - self.window_size
            X_win = self.X_all[start:last_t]
            y_win = self.y_all[start:last_t]
            # instantiate and fit model once to inspect core tensor shapes
            if method == "hopls_milr":
                from .hopls_milr import HOPLS_MILR

                model_ins = HOPLS_MILR(
                    **{
                        k: method_kwargs[k]
                        for k in ["R", "Ln", "epsilon", "lambda_X", "lambda_Y", "alpha"]
                    }
                )
            else:
                from .hopls import HOPLS as HOPLS_INS

                model_ins = HOPLS_INS(
                    **{k: method_kwargs[k] for k in ["R", "Ln", "epsilon"]}
                )
            import torch

            model_ins.fit(torch.Tensor(X_win), torch.Tensor(y_win))
            _print_hopls_shapes(model_ins)

        return y_pred_all, y_true_all, time_index_test, metrics

    def run_split(
        self,
        train_split: float = 0.8,
        method: str = "hopls",
        verbose: bool = False,
        **method_kwargs: Any,
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
            "lightgbm": lightgbm_predictor,
            "hopls_milr_rhooi": hopls_milr_rhooi_predictor,
        }
        mode_map = {
            "hopls": "tensor",
            "ridge": "matrix",
            "hopls_ridge": "tensor",
            "linear_regression": "matrix",
            "hopls_milr": "tensor",
            "lightgbm": "matrix",
            "hopls_milr_rhooi": "tensor",
        }

        if method not in predictor_map:
            raise ValueError(
                f"Unknown method '{method}', available: {list(predictor_map.keys())}"
            )

        predictor = predictor_map[method]
        mode = mode_map[method]

        # Calculate split point
        split_idx = int(train_split * self.T)
        if split_idx <= 0 or split_idx >= self.T:
            raise ValueError(
                f"Invalid train_split {train_split}, results in split_idx={split_idx} for T={self.T}"
            )

        if verbose:
            print(
                f"Using train/test split: {split_idx}/{self.T - split_idx} (train/test)"
            )

        # Split data
        # shape (train_size, n_series, n_features)
        X_train = self.X_all[:split_idx]
        y_train = self.y_all[:split_idx]  # shape (train_size, n_series)
        # shape (test_size, n_series, n_features)
        X_test = self.X_all[split_idx:]
        y_true = self.y_all[split_idx:]  # shape (test_size, n_series)

        if mode == "matrix":
            # flatten training data: (train_size * n_series) × n_features
            X_tr = X_train.reshape(-1, self.n_features)
            y_tr = y_train.reshape(-1)
            # flatten test data: (test_size * n_series) × n_features
            X_te = X_test.reshape(-1, self.n_features)

            if verbose:
                print(
                    f"Matrix mode: X_train shape {X_tr.shape}, y_train shape {y_tr.shape}"
                )
                print(f"Matrix mode: X_test shape {X_te.shape}")

            y_pred_flat = predictor(X_tr, y_tr, X_te, **method_kwargs)
            # reshape back to (test_size, n_series)
            y_pred = y_pred_flat.reshape(X_test.shape[0], X_test.shape[1])

        else:  # tensor mode
            if verbose:
                print(
                    f"Tensor mode: X_train shape {X_train.shape}, y_train shape {y_train.shape}"
                )
                print(f"Tensor mode: X_test shape {X_test.shape}")

            y_pred = predictor(X_train, y_train, X_test, **method_kwargs)
            y_pred = np.asarray(y_pred)

        # Time index for test set
        time_index_test = (
            self.time_index[split_idx:] if self.time_index is not None else None
        )

        # Compute metrics
        mse = mean_squared_error(y_true.ravel(), y_pred.ravel())
        r2 = r2_score(y_true.ravel(), y_pred.ravel())
        true_dir = np.sign(y_true.ravel())
        pred_dir = np.sign(y_pred.ravel())
        mask = true_dir != 0
        directional_acc = (
            np.mean(pred_dir[mask] == true_dir[mask]) if mask.any() else np.nan
        )
        metrics = {"mse": mse, "r2": r2,
                   "directional_accuracy": directional_acc}

        # Store for plotting
        self.y_pred_all = y_pred
        self.y_true_all = y_true
        self.time_index_test = time_index_test
        self.metrics = metrics

        if verbose:
            print(
                f"Split results - MSE: {mse:.6f}, R2: {r2:.6f}, Dir. Acc.: {directional_acc:.6f}"
            )

        return y_pred, y_true, time_index_test, metrics

    def _predict_single(
        self, t: int, predictor: Callable, mode: str, kw: Dict[str, Any]
    ) -> np.ndarray:
        """
        Slice out the window at time t and call the predictor.
        """
        start = t - self.window_size
        # shape (window_size, n_series, n_features)
        X_win = self.X_all[start:t]
        y_win = self.y_all[start:t]  # shape (window_size, n_series)

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
            # add leading dim → (1, series, features)
            X_te = self.X_all[t][None, ...]
            y_pred = predictor(X_tr, y_tr, X_te, **kw)
            return np.asarray(y_pred).squeeze(0)

    def plot_results(self, series_indices: Optional[Sequence[int]] = None):
        """
        Plot true vs predicted for selected series, two ways:
          1. Predicted shifted one step ahead (convention)
          2. Predicted aligned back to true values (for magnitude comparison)
        """
        if not hasattr(self, "y_pred_all"):
            raise RuntimeError("No predictions found. Run run() first.")
        y_pred = self.y_pred_all
        y_true = self.y_true_all
        times = (
            np.asarray(self.time_index_test)
            if self.time_index_test is not None
            else np.arange(len(y_true))
        )
        if series_indices is None:
            series_indices = list(range(min(3, self.n_series)))

        # 1. shifted one step ahead
        plt.figure(figsize=(12, 6))
        for idx in series_indices:
            plt.plot(times, y_true[:, idx], label=f"Actual series {idx}")
            plt.plot(
                times, y_pred[:, idx], "--", label=f"Predicted (shifted) series {idx}"
            )
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
            plt.plot(aligned_times, y_true[:-1, idx],
                     label=f"Actual series {idx}")
            plt.plot(
                aligned_times,
                aligned_pred[:-1, idx],
                "--",
                label=f"Predicted aligned series {idx}",
            )
        plt.xlabel("Time")
        plt.ylabel("Target value")
        plt.title("Predicted vs Actual (aligned back)")
        plt.legend()
        plt.show()

    def summary(self) -> Dict[str, Any]:
        """Return summary of test metrics."""
        if not hasattr(self, "metrics"):
            raise RuntimeError("No metrics found. Run run() first.")
        return self.metrics

    def run_comparison(
        self,
        methods_params: Dict[str, Dict[str, Any]],
        n_jobs: int = 1,
        verbose: bool = False,
        series_indices: Optional[Sequence[int]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run and compare multiple prediction methods with given parameters, print metrics,
        and plot actual vs predicted for specified series on subplots.
        """
        import matplotlib.pyplot as plt

        results: Dict[str, Dict[str, Any]] = {}
        for method, params in methods_params.items():
            y_pred, y_true, times, metrics = self.run_window(
                method=method, verbose=verbose, n_jobs=n_jobs, **params
            )
            results[method] = {
                "y_pred": y_pred,
                "y_true": y_true,
                "times": times,
                "metrics": metrics,
            }
            print(f"Metrics for {method}: {metrics}")
        # default series indices to first two if not provided
        if series_indices is None:
            series_indices = [0, 1]
        # plot comparisons in one figure per method
        fig, axes = plt.subplots(
            nrows=len(results), ncols=1, figsize=(12, 6 * len(results)), sharex=True
        )
        if len(results) == 1:
            axes = [axes]
        for ax, (method, res) in zip(axes, results.items()):
            y_pred = res["y_pred"]
            y_true = res["y_true"]
            times = res["times"]
            for idx in series_indices:
                ax.plot(times, y_true[:, idx], label=f"Actual series {idx}")
                ax.plot(
                    times,
                    y_pred[:, idx],
                    "--",
                    label=f"{method} predicted series {idx}",
                )
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
    if hasattr(model, "model"):
        P_list, Q_mat, D, T_mat, W_mat = model.model
        print(f"Components: {len(P_list)}")
        print("P shapes per component:")
        for i, Pr in enumerate(P_list):
            print(f"  Comp {i}: {[tuple(p.shape) for p in Pr]}")
        print(f"Q_mat shape: {tuple(Q_mat.shape)}, D shape: {tuple(D.shape)}")
        print(
            f"T_mat shape: {tuple(T_mat.shape)}, W_mat shape: {tuple(W_mat.shape)}")
        return
    # HOPLS-MILR
    if hasattr(model, "P_r_all_components"):
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

# Added imports for PyPortfolioOpt
from pypfopt import EfficientFrontier, risk_models
from pypfopt.hierarchical_portfolio import HRPOpt # Added for HRP

def run_rolling_portfolio_backtest(
    # Data for PredictionTestEngine & Portfolio Construction
    X_features: np.ndarray,
    y_returns_full: np.ndarray, # Used by engine.run_window() and for portfolio construction inputs
    pred_engine_window_size: int,
    asset_names: list[str],
    hist_data_lookback: int, # Renamed from mvo_cov_lookback for generality

    # Prediction Configuration
    pred_method_name: str = "hopls",
    pred_method_kwargs: Optional[Dict[str, Any]] = None,
    pred_engine_train_start: Optional[int] = None,
    pred_engine_train_end: Optional[int] = None,
    pred_engine_time_index: Optional[Sequence] = None,
    pred_engine_n_jobs: int = 1,

    # Portfolio Construction Configuration
    portfolio_optimizer_method: str = "mvo", # "mvo", "min_volatility", "equal_weighting", "hrp"
    rebalance_freq: int = 1, # Renamed from mvo_rebalance_freq

    # MVO Specific Configuration (used if portfolio_optimizer_method == "mvo")
    mvo_risk_free_rate_per_period: float = 0.0,
    mvo_target_objective: str = 'max_sharpe', # e.g., 'max_sharpe', 'min_volatility' (within MVO context), 'efficient_risk', 'efficient_return'
    mvo_target_volatility_per_period: Optional[float] = None,
    mvo_target_return_per_period: Optional[float] = None,
    
    # Common for MVO and Min_Volatility (EfficientFrontier based)
    ef_max_asset_weight: float = 1.0, # Renamed from mvo_max_asset_weight
    ef_min_asset_weight: float = 0.0, # Renamed from mvo_min_asset_weight
    ef_solver: Optional[str] = None, # Renamed from mvo_solver
    
    rank_k: int = 10,  # Number of assets to select for simple ranking strategies
    verbose: bool = False
) -> pd.DataFrame:
    """
    Performs a rolling window portfolio backtest using a specified prediction method
    and portfolio construction strategy.

    Args:
        X_features: 3D Numpy array of features (T, n_series, n_features) for PredictionTestEngine.
        y_returns_full: 2D Numpy array of all historical actual returns (T, n_series).
                        Used by PredictionTestEngine and for portfolio construction inputs.
        pred_engine_window_size: Window size for the PredictionTestEngine.
        asset_names: List of strings for asset names.
        hist_data_lookback: Number of past periods of actual returns for historical data inputs
                            (e.g., covariance matrix for MVO/MinVol, returns for HRP).

        pred_method_name: Name of the prediction method to use in PredictionTestEngine.
        pred_method_kwargs: Dictionary of keyword arguments for the prediction method.
        pred_engine_train_start: Optional start index for training in PredictionTestEngine.
        pred_engine_train_end: Optional end index for training in PredictionTestEngine.
        pred_engine_time_index: Optional time index for data in PredictionTestEngine.
        pred_engine_n_jobs: Number of parallel jobs for PredictionTestEngine's run_window.

        portfolio_optimizer_method: Strategy for portfolio construction.
                                    Options: "mvo", "min_volatility", "equal_weighting", "hrp".
        rebalance_freq: How often to rebalance the portfolio (in periods).

        mvo_risk_free_rate_per_period: Risk-free rate for MVO 'max_sharpe' (per period).
        mvo_target_objective: MVO objective (e.g., 'max_sharpe', 'min_volatility', 'efficient_risk', 'efficient_return').
                              Note: 'min_volatility' here is an MVO objective; for the dedicated minimum volatility
                              portfolio without predicted returns, use portfolio_optimizer_method="min_volatility".
        mvo_target_volatility_per_period: Target volatility for MVO 'efficient_risk'.
        mvo_target_return_per_period: Target return for MVO 'efficient_return'.
        
        ef_max_asset_weight: Maximum asset weight for EfficientFrontier-based methods (MVO, MinVol).
        ef_min_asset_weight: Minimum asset weight for EfficientFrontier-based methods (MVO, MinVol).
        ef_solver: Solver for CVXPY used by EfficientFrontier (MVO, MinVol).
        
        rank_k: Number of assets to use for 'top_k' or 'long_short' strategies.
        verbose: If True, print progress and warnings.

    Returns:
        A pandas DataFrame with portfolio returns and asset weights over the backtest period.
    """

    # 1. Instantiate PredictionTestEngine
    if verbose:
        print("Initializing PredictionTestEngine...")
    # Determine training window end (defaults to end of data)
    engine_train_end = pred_engine_train_end if pred_engine_train_end is not None else X_features.shape[0]
    engine = PredictionTestEngine(
        X_all=X_features,
        y_all=y_returns_full,
        window_size=pred_engine_window_size,
        train_start=pred_engine_train_start,
        train_end=engine_train_end,
        time_index=pred_engine_time_index
    )
    if verbose:
        print(f"PredictionTestEngine initialized. Train start index: {engine.train_start}, Test indices count: {len(engine.test_indices)}")

    # 2. Generate predictions (expected returns for MVO, or can be ignored by other methods)
    if verbose:
        print(f"Running prediction method: {pred_method_name} to get expected returns (mu)...")
    
    predicted_returns_for_mvo, actual_returns_for_pnl, time_index_for_results, pred_metrics = engine.run_window(
        method=pred_method_name,
        verbose=verbose, # Pass verbose to sub-methods if they support it
        n_jobs=pred_engine_n_jobs,
        **(pred_method_kwargs if pred_method_kwargs else {})
    )
    if verbose:
        print(f"Predictions generated. Shape: {predicted_returns_for_mvo.shape}. Metrics: {pred_metrics}")

    if predicted_returns_for_mvo.size == 0:
        if verbose:
            print("Warning: Predictor returned no predictions. Returning empty DataFrame.")
        cols = ['portfolio_return'] + [f'weight_{asset}' for asset in asset_names]
        idx = time_index_for_results if time_index_for_results is not None else pd.RangeIndex(start=0, stop=0, step=1)
        return pd.DataFrame(columns=cols, index=idx)

    # 3. Portfolio Construction Core Logic
    portfolio_construction_test_period_start_idx_in_full_data = engine.train_start # Where test period begins in y_returns_full

    num_test_periods, num_assets = predicted_returns_for_mvo.shape
    if len(asset_names) != num_assets:
        raise ValueError("Length of asset_names must match number of assets in predicted/actual return arrays.")
    if time_index_for_results is not None and len(time_index_for_results) != num_test_periods:
        raise ValueError("Length of time_index_for_results must match num_test_periods.")

    portfolio_log = []
    # Initialize with equal weights as a safe default or for first period before rebalance
    last_successful_weights = {asset: 1.0 / num_assets for asset in asset_names}

    if verbose:
        print(f"Starting rolling backtest with portfolio optimizer: {portfolio_optimizer_method}...")

    for t in tqdm(range(num_test_periods), desc=f"Rolling Backtest ({portfolio_optimizer_method})", disable=not verbose):
        current_weights = last_successful_weights # Default to previous weights if rebalance fails or not scheduled

        if t % rebalance_freq == 0:
            # Prepare historical data for covariance/HRP
            # Data up to *before* current period t's decision point
            hist_data_end_idx = min(portfolio_construction_test_period_start_idx_in_full_data + t, y_returns_full.shape[0])
            hist_data_start_idx = max(0, hist_data_end_idx - hist_data_lookback)

            if hist_data_start_idx < 0:
                if verbose and t > 0: # Avoid warning at t=0 if lookback is larger than initial history
                    print(f"  Warning (t={t}): Historical lookback extends before start of data. Using available history from index 0.")
                hist_data_start_idx = 0
            
            # Default to previous weights if not enough data for the chosen method
            perform_rebalance = True
            if hist_data_end_idx <= hist_data_start_idx or hist_data_end_idx == 0:
                if verbose:
                    print(f"  Warning (t={t}): Not enough historical data (end_idx={hist_data_end_idx}, start_idx={hist_data_start_idx}). Using previous weights.")
                perform_rebalance = False
            if not perform_rebalance and verbose:
                print(f"  Debug (t={t}): skip rebalance, hist_data_start_idx={hist_data_start_idx}, hist_data_end_idx={hist_data_end_idx}")
            
            historical_returns_for_opt_df = None
            if perform_rebalance and portfolio_optimizer_method in ["mvo", "min_volatility", "hrp"]:
                historical_returns_for_opt_np = y_returns_full[hist_data_start_idx:hist_data_end_idx, :]
                historical_returns_for_opt_df = pd.DataFrame(historical_returns_for_opt_np, columns=asset_names)
                if historical_returns_for_opt_df.shape[0] < 2 : # Min rows for cov/HRP
                    if verbose:
                        print(f"  Warning (t={t}): Insufficient rows ({historical_returns_for_opt_df.shape[0]}) in historical data for {portfolio_optimizer_method}. Using previous weights.")
                    perform_rebalance = False

            if perform_rebalance:
                try:
                    new_weights = None
                    # Traditional optimizers
                    if portfolio_optimizer_method == "mvo":
                        mu_series = pd.Series(predicted_returns_for_mvo[t, :], index=asset_names)
                        S_matrix = risk_models.sample_cov(historical_returns_for_opt_df, frequency=1)
                        ef = EfficientFrontier(mu_series, S_matrix, weight_bounds=(ef_min_asset_weight, ef_max_asset_weight), solver=ef_solver)
                        
                        if mvo_target_objective == 'max_sharpe':
                            ef.max_sharpe(risk_free_rate=mvo_risk_free_rate_per_period)
                        elif mvo_target_objective == 'min_volatility': # MVO's own min_volatility (can use mu)
                            ef.min_volatility()
                        elif mvo_target_objective == 'efficient_risk':
                            if mvo_target_volatility_per_period is None: raise ValueError("mvo_target_volatility_per_period needed for 'efficient_risk'.")
                            ef.efficient_risk(target_volatility=mvo_target_volatility_per_period)
                        elif mvo_target_objective == 'efficient_return':
                            if mvo_target_return_per_period is None: raise ValueError("mvo_target_return_per_period needed for 'efficient_return'.")
                            ef.efficient_return(target_return=mvo_target_return_per_period)
                        else:
                            raise ValueError(f"Unsupported mvo_target_objective: {mvo_target_objective}")
                        new_weights = ef.clean_weights()

                    elif portfolio_optimizer_method == "min_volatility":
                        S_matrix = risk_models.sample_cov(historical_returns_for_opt_df, frequency=1)
                        ef = EfficientFrontier(None, S_matrix, weight_bounds=(ef_min_asset_weight, ef_max_asset_weight), solver=ef_solver)
                        ef.min_volatility() # Dedicated min volatility portfolio (mu=None)
                        new_weights = ef.clean_weights()

                    elif portfolio_optimizer_method == "hrp":
                        hrp_opt = HRPOpt(historical_returns_for_opt_df)
                        new_weights = hrp_opt.optimize()  # Returns a dict of weights
                    # Simple ranking strategies
                    elif portfolio_optimizer_method == "top_k":
                        preds_row = predicted_returns_for_mvo[t, :]
                        sorted_idx = np.argsort(preds_row)[::-1]
                        top_idx = sorted_idx[:rank_k]
                        new_weights = {asset_names[j]: 1.0/rank_k for j in top_idx}
                    elif portfolio_optimizer_method == "long_short":
                        preds_row = predicted_returns_for_mvo[t, :]
                        sorted_idx = np.argsort(preds_row)
                        bottom_idx = sorted_idx[:rank_k]
                        top_idx = sorted_idx[::-1][:rank_k]
                        new_weights = {}
                        # Long top_k and short bottom_k with dollar-neutral exposure
                        for j in top_idx:
                            new_weights[asset_names[j]] = 0.5/rank_k
                        for j in bottom_idx:
                            new_weights[asset_names[j]] = -0.5/rank_k
                    elif portfolio_optimizer_method == "equal_weighting":
                        new_weights = {asset: 1.0 / num_assets for asset in asset_names}
                    
                    else:
                        raise ValueError(f"Unsupported portfolio_optimizer_method: {portfolio_optimizer_method}")

                    if new_weights: # Ensure weights were actually computed
                        current_weights = new_weights
                        last_successful_weights = new_weights

                except ValueError as ve: # Catch config errors or PyPortfolioOpt value errors
                    if verbose: print(f"  ValueError (t={t}) during {portfolio_optimizer_method} opt: {ve}. Using previous weights.")
                except Exception as e: # Catch other errors like solver issues
                    if verbose: print(f"  General error (t={t}) during {portfolio_optimizer_method} opt: {e}. Using previous weights.")
        
        # Calculate portfolio return for period t using actual returns for that period
        actual_period_asset_returns = actual_returns_for_pnl[t, :]
        portfolio_return_for_period_t = sum(current_weights.get(asset_names[j], 0) * actual_period_asset_returns[j] for j in range(num_assets))

        log_entry = {'portfolio_return': portfolio_return_for_period_t}
        for asset_name_val in asset_names:
            log_entry[f'weight_{asset_name_val}'] = current_weights.get(asset_name_val, 0.0)
        portfolio_log.append(log_entry)

    if not portfolio_log:
        cols = ['portfolio_return'] + [f'weight_{asset}' for asset in asset_names]
        idx = time_index_for_results if time_index_for_results is not None else pd.RangeIndex(start=0, stop=0, step=1)
        return pd.DataFrame(columns=cols, index=idx)

    results_df = pd.DataFrame(portfolio_log)
    if time_index_for_results is not None:
        results_df.index = time_index_for_results
    else:
        results_df.index.name = 'period_index_in_test' # Or some other meaningful default
        
    if verbose:
        print(f"Rolling backtest with {portfolio_optimizer_method} completed.")
    return results_df

def plot_portfolio_performance(
    results_df: pd.DataFrame,
    asset_names: list[str],
    title_suffix: str = ""
):
    """
    Plots the cumulative portfolio return and, optionally, asset weights over time.
    Also calculates and displays Sharpe Ratio and Maximum Drawdown.
    """
    plt.style.use('seaborn-v0_8-darkgrid') # Using a seaborn style

    fig, ax1 = plt.subplots(figsize=(10, 5)) # Changed figsize

    # Calculate portfolio returns (daily or monthly, depending on data frequency)
    # Assuming 'portfolio_return' column contains per-period returns (not cumulative)
    portfolio_returns = results_df['portfolio_return']

    # Plot cumulative portfolio return
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    ax1.plot(cumulative_returns.index, cumulative_returns, 
             label='Cumulative Portfolio Return', color='dodgerblue', linewidth=2)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Cumulative Return', color='dodgerblue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='dodgerblue')
    ax1.set_title(f'Portfolio Performance {title_suffix}', fontsize=14)
    ax1.grid(True, which='major', linestyle='--', alpha=0.7)

    # Calculate Sharpe Ratio (annualized, assuming monthly returns for now, risk-free rate = 0)
    # Adjust N if returns are daily (N=252) or other frequency
    N = 12 # Annualization factor for monthly returns
    sharpe_ratio = np.nan # Default to NaN if calculation is not possible
    if not portfolio_returns.empty and portfolio_returns.std() != 0:
        sharpe_ratio = (portfolio_returns.mean() * N) / (portfolio_returns.std() * np.sqrt(N))

    # Calculate Maximum Drawdown
    max_drawdown = np.nan # Default to NaN
    if not cumulative_returns.empty:
        # Calculate the running maximum
        running_max = (cumulative_returns + 1).cummax()
        # Calculate the drawdown
        drawdown = (cumulative_returns + 1) / running_max - 1
        max_drawdown = drawdown.min()

    # Add Sharpe Ratio and Max Drawdown to the plot
    stats_text = f"Annualised Sharpe Ratio: {sharpe_ratio:.2f}, Maximum Drawdown: {max_drawdown:.2%}"
    # Position the text box. Adjust x, y, ha, va as needed for your plot.
    ax1.text(0.02, 0.02, stats_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    
    # # Plot asset weights over time on a second y-axis
    # # ax2 = ax1.twinx() 
    # # weight_cols = [f\'weight_{asset}\' for asset in asset_names]
    # # for col in weight_cols:
    # #     ax2.plot(results_df.index, results_df[col], label=col.replace("weight_", "W: "), alpha=0.6, linestyle=\'--\')
    # # ax2.set_ylabel(\'Asset Weights\', color=\'gray\')
    # # ax2.tick_params(axis=\'y\', labelcolor=\'gray\')
    # # ax2.set_ylim(0, results_df[weight_cols].max().max() * 1.1 if not results_df[weight_cols].empty else 1) # Adjust y-limit for weights

    fig.tight_layout() # otherwise the right y-label is slightly clipped
    
    # Combine legends if both plots were active
    # For now, only ax1 legend is needed.
    lines, labels = ax1.get_legend_handles_labels()
    # if \'ax2\' in locals(): # If asset weights were plotted
    #     lines2, labels2 = ax2.get_legend_handles_labels()
    #     ax1.legend(lines + lines2, labels + labels2, loc=\'upper left\')
    # else:
    ax1.legend(lines, labels, loc='upper left', fontsize=11)

    plt.show()

def backtest_from_predictions(
    predicted_returns_for_mvo: np.ndarray,
    actual_returns_for_pnl: np.ndarray,
    time_index_for_results: Optional[Sequence],
    asset_names: list[str],
    hist_data_lookback: int,
    portfolio_optimizer_method: str = "top_k",
    rank_k: int = 10,
    rebalance_freq: int = 1,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Run portfolio backtest given precomputed predictions and actuals.
    """
    num_test_periods, num_assets = predicted_returns_for_mvo.shape
    if len(asset_names) != num_assets:
        raise ValueError("Length of asset_names must match number of assets.")
    if time_index_for_results is not None and len(time_index_for_results) != num_test_periods:
        raise ValueError("Length of time_index_for_results must match number of periods.")
    portfolio_log = []
    last_weights = {asset: 1.0/num_assets for asset in asset_names}
    for t in range(num_test_periods):
        current_weights = last_weights
        if t % rebalance_freq == 0:
            start_idx = max(0, t - hist_data_lookback)
            end_idx = t
            if end_idx - start_idx >= 2:
                # determine new weights based on simple ranking strategies
                preds = predicted_returns_for_mvo[t]
                if portfolio_optimizer_method == 'top_k':
                    top_idx = np.argsort(preds)[::-1][:rank_k]
                    current_weights = {asset: 0.0 for asset in asset_names}
                    for i in top_idx:
                        current_weights[asset_names[i]] = 1.0/rank_k
                elif portfolio_optimizer_method == 'long_short':
                    sorted_idx = np.argsort(preds)
                    bottom_idx = sorted_idx[:rank_k]
                    top_idx = sorted_idx[::-1][:rank_k]
                    current_weights = {asset: 0.0 for asset in asset_names}
                    for i in top_idx:
                        current_weights[asset_names[i]] = 0.5/rank_k
                    for i in bottom_idx:
                        current_weights[asset_names[i]] = -0.5/rank_k
                elif portfolio_optimizer_method == 'equal_weighting':
                    current_weights = {asset: 1.0/num_assets for asset in asset_names}
                else:
                    raise ValueError(f"Unsupported method: {portfolio_optimizer_method}")
