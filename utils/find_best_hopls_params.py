import json
import pandas as pd
import numpy as np
from time import time
from itertools import product
from datetime import datetime
import matplotlib.pyplot as plt
from model.prediction_engine import PredictionTestEngine
import os

# Change working directory to the script's parent directory (project root)
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

    # Hyperparameter grid
    window_sizes = [50, 60, 70, 80]
    R_values = [10, 20, 30, 40, 60, 80, 100]
    Ln_values = [(2,2), (4,4), (8,8), (15,15), (20,20), (25,25)]
    configs = list(product(window_sizes, R_values, Ln_values))

    # Prepare logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/gridsearch_log_{timestamp}.txt"
    log_file = open(log_path, 'w', buffering=1)  # line buffering

    total = len(configs)
    print(f"Starting grid search ({total} combinations)...", file=log_file)
    start_time = time()

    results = []
    for idx, (window_size, R, Ln) in enumerate(configs, 1):
        print(f"[{idx}/{total}] window_size={window_size}, R={R}, Ln={Ln}", file=log_file)
        try:
            engine = PredictionTestEngine(
                X_all, Y_all,
                window_size=window_size,
                time_index=time_index_all
            )
            _, _, _, metrics = engine.run(
                method="hopls_ridge",
                R=R,
                Ln=Ln,
                epsilon=1e-8,
                verbose=False,
                n_jobs=1
            )
            res = {
                'window_size': window_size,
                'R': R,
                'Ln': Ln,
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
                'status': 'error',
                'error_message': str(e)
            }
            results.append(err)
            print(f"  ERROR: {e}", file=log_file)

    elapsed = time() - start_time
    print(f"Grid search finished in {elapsed:.2f}s", file=log_file)

    # Save JSON results
    json_path = f"logs/gridsearch_results_{timestamp}.json"
    with open(json_path, 'w') as jf:
        json.dump(results, jf, indent=2)
    print(f"Results saved to {json_path}", file=log_file)

    log_file.close()
    print(f"Log written to {log_path}")

if __name__ == "__main__":
    main()