\
import json
import pandas as pd
import numpy as np
from time import time
from itertools import product
from datetime import datetime
import matplotlib.pyplot as plt

# ensure project root on path for model import
import os
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

from model.prediction_engine import PredictionTestEngine

def main():
    # Load data
    final_data = pd.read_csv('data/final_data.csv')
    final_data = final_data.sort_values(['year_month', 'GVKEY'])
    final_data['trt1m'] = pd.to_numeric(final_data['trt1m'], errors='coerce')

    # Feature columns
    feature_cols = [c for c in final_data.columns 
                    if c not in ['GVKEY','year_month','trt1m']]
    for col in feature_cols:
        final_data[col] = pd.to_numeric(final_data[col], errors='coerce')

    # Build Y
    Y_df = (final_data
        .pivot_table(index='year_month', columns='GVKEY', 
                     values='trt1m', aggfunc='mean')
        .sort_index().sort_index(axis=1)
    )
    Y_df = Y_df.apply(lambda x: (x - x.mean())/x.std(), axis=0)
    Y = Y_df.to_numpy()

    # Build X tensor
    X_list = []
    for col in feature_cols:
        pivot = (final_data
            .pivot_table(index='year_month', columns='GVKEY', 
                         values=col, aggfunc='mean')
            .reindex(index=Y_df.index, columns=Y_df.columns)
        )
        pivot = pivot.apply(lambda x: (x - x.mean())/x.std(), axis=0)
        X_list.append(pivot.to_numpy())
    X = np.stack(X_list, axis=2)

    # Shift data
    X_all = np.nan_to_num(X[:-1, :, :])
    Y_all = np.nan_to_num(Y[1:, :])
    time_index_all = pd.to_datetime(Y_df.index[1:], format='%Y-%m')

    # Hyperparameter grid for hopls_milr_rhooi
    window_sizes = [50]
    R_values = [15]
    Ln_values = [(15, 15), (20, 20)]
    lambda_XY_values = [0.075]  # For lambda_X and lambda_Y
    alpha_values = [3, 5, 7]
    lambda_P_factor_penalty_values = [0.001]
    lambda_Q_factor_penalty_values = [0.001]
    rhooi_n_iter_max_values = [100, 150]
    rhooi_tol_values = [1e-08, 1e-09]

    configs = list(product(
        window_sizes, R_values, Ln_values, lambda_XY_values, alpha_values,
        lambda_P_factor_penalty_values, lambda_Q_factor_penalty_values,
        rhooi_n_iter_max_values, rhooi_tol_values
    ))

    # Prepare logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/gridsearch_hopls_milr_rhooi_log_{timestamp}.txt"
    log_file = open(log_path, 'w', buffering=1)  # line buffering

    total = len(configs)
    print(f"Starting grid search for hopls_milr_rhooi ({total} combinations)...", file=log_file)
    start_time = time()

    results = []
    epsilon = 1e-7 # Based on provided successful runs
    test_start_percentage = 0.5 # Updated test_start_percentage
    n_jobs_val = 7 # Defaulting to 7 as seen in notebook
    rhooi_verbose_val = False # Defaulting to False for grid search

    print(f"Using epsilon: {epsilon}", file=log_file)
    print(f"Test start percentage: {test_start_percentage}", file=log_file)
    print(f"Number of jobs for engine: {n_jobs_val}", file=log_file)
    print(f"Rhooi verbose: {rhooi_verbose_val}", file=log_file)
    print('\\n' ,file=log_file)
    
    for idx, (
        window_size, R, Ln, lambda_XY, alpha,
        lambda_P_factor_penalty, lambda_Q_factor_penalty,
        rhooi_n_iter_max, rhooi_tol
    ) in enumerate(configs, 1):
        print(f"[{idx}/{total}] window_size={window_size}, R={R}, Ln={Ln}, lambda_XY={lambda_XY}, alpha={alpha}, "
              f"lambda_P_penalty={lambda_P_factor_penalty}, lambda_Q_penalty={lambda_Q_factor_penalty}, "
              f"rhooi_n_iter_max={rhooi_n_iter_max}, rhooi_tol={rhooi_tol}", file=log_file)
        try:
            engine = PredictionTestEngine(
                X_all, Y_all,
                window_size=window_size,
                train_start = int(test_start_percentage * X_all.shape[0]),
                time_index=time_index_all
            )
            _, _, _, metrics = engine.run_window(
                method="hopls_milr_rhooi",
                R=R,
                Ln=Ln,
                epsilon=epsilon,
                verbose=False, # Keep verbose False for grid search, use True for individual runs
                n_jobs=n_jobs_val,
                lambda_X=lambda_XY,
                lambda_Y=lambda_XY,
                alpha=alpha,
                lambda_P_factor_penalty=lambda_P_factor_penalty,
                lambda_Q_factor_penalty=lambda_Q_factor_penalty,
                rhooi_verbose=rhooi_verbose_val,
                rhooi_n_iter_max=rhooi_n_iter_max,
                rhooi_tol=rhooi_tol
            )
            res = {
                'window_size': window_size,
                'R': R,
                'Ln': Ln,
                'lambda_XY': lambda_XY,
                'alpha': alpha,
                'lambda_P_factor_penalty': lambda_P_factor_penalty,
                'lambda_Q_factor_penalty': lambda_Q_factor_penalty,
                'rhooi_n_iter_max': rhooi_n_iter_max,
                'rhooi_tol': rhooi_tol,
                'mse': metrics['mse'],
                'r2': metrics['r2'],
                'directional_accuracy': metrics['directional_accuracy'],
                'status': 'success'
            }
            results.append(res)
            print(f"  OK: mse={res['mse']:.6f}, r2={res['r2']:.6f}, dir_acc={res['directional_accuracy']:.6f}", 
                  file=log_file)
        except Exception as e:
            err = {
                'window_size': window_size,
                'R': R,
                'Ln': Ln,
                'lambda_XY': lambda_XY,
                'alpha': alpha,
                'lambda_P_factor_penalty': lambda_P_factor_penalty,
                'lambda_Q_factor_penalty': lambda_Q_factor_penalty,
                'rhooi_n_iter_max': rhooi_n_iter_max,
                'rhooi_tol': rhooi_tol,
                'status': 'error',
                'error_message': str(e)
            }
            results.append(err)
            print(f"  ERROR: {e}", file=log_file)

    elapsed = time() - start_time
    print(f"Grid search for hopls_milr_rhooi finished in {elapsed:.2f}s", file=log_file)

    # Save JSON results
    json_path = f"logs/gridsearch_hopls_milr_rhooi_results_{timestamp}.json"
    with open(json_path, 'w') as jf:
        json.dump(results, jf, indent=2)
    print(f"Results saved to {json_path}", file=log_file)

    log_file.close()
    print(f"Log written to {log_path}")

if __name__ == "__main__":
    main()
