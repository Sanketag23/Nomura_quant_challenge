import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Union
import task3
_TAUS = [5, 10, 15, 20, 25, 30]

def _compute_internal_pnl(df: pd.DataFrame, tau: int) -> np.ndarray:
    side = df['Side'].astype(float).values
    vol = df['Volume'].astype(float).values
    mtau = df[f'M{tau}'].astype(float).values
    tp = df['Trade Price'].astype(float).values
    return side * vol * (mtau - tp)

def optimal_threshold(df_val: pd.DataFrame, df_test: pd.DataFrame, tau: int, use_client_specific: bool=True) -> dict:
    p_val = task3.predict_adversity(df=df_val, tau=tau)
    p_test = task3.predict_adversity(df=df_test, tau=tau)
    pnl_val_int = _compute_internal_pnl(df_val, tau)
    pnl_test_int = _compute_internal_pnl(df_test, tau)
    clients = df_val['Name'].unique()
    thetas_to_search = np.linspace(0, 1, 101)
    if not use_client_specific:
        best_pnl = -np.inf
        best_th = 0.0
        for th in thetas_to_search:
            pnl = np.where(p_val > th, 0, pnl_val_int).sum()
            if pnl > best_pnl:
                best_pnl = pnl
                best_th = th
        test_pnl = np.where(p_test > best_th, 0, pnl_test_int).sum()
        return {'theta': best_th, 'validation_pnl': best_pnl, 'test_pnl': test_pnl}
    else:
        theta_dict = {}
        val_pnl_total = 0.0
        test_pnl_total = 0.0
        for c in clients:
            mask_val = (df_val['Name'] == c).values
            mask_test = (df_test['Name'] == c).values
            p_val_c = p_val[mask_val]
            pnl_val_int_c = pnl_val_int[mask_val]
            best_pnl = -np.inf
            best_th = 0.0
            for th in thetas_to_search:
                pnl = np.where(p_val_c > th, 0, pnl_val_int_c).sum()
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_th = th
            theta_dict[c] = best_th
            val_pnl_total += best_pnl
            p_test_c = p_test[mask_test]
            pnl_test_int_c = pnl_test_int[mask_test]
            test_pnl_c = np.where(p_test_c > best_th, 0, pnl_test_int_c).sum()
            test_pnl_total += test_pnl_c
        return {'theta': theta_dict, 'validation_pnl': val_pnl_total, 'test_pnl': test_pnl_total}

def plot_all_pnl_vs_theta(df_val: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 7))
    thetas = np.linspace(0, 1, 101)
    for tau in _TAUS:
        p_val = task3.predict_adversity(df=df_val, tau=tau)
        pnl_val_int = _compute_internal_pnl(df_val, tau)
        pnls = []
        for th in thetas:
            pnl = np.where(p_val > th, 0, pnl_val_int).sum()
            pnls.append(pnl)
        plt.plot(thetas, pnls, label=f'τ={tau}', linewidth=2)
    plt.xlabel('Threshold (θ)')
    plt.ylabel('Validation PnL')
    plt.title('Validation PnL vs Externalization Threshold (All Horizons)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    here = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(here, 'pnl_vs_theta.png'))
    plt.close()

def main():
    df = task3._load_raw()
    tr_idx, va_idx, te_idx = task3._split_indices(df)
    df_val = df.iloc[va_idx].copy()
    df_test = df.iloc[te_idx].copy()
    results_rows = []
    print('=' * 60)
    print(f'{'Tau':>4} | {'Baseline (No Ext.)':>20} | {'Externalized PnL':>20}')
    print('-' * 60)
    plot_all_pnl_vs_theta(df_val)
    for tau in _TAUS:
        pnl_test_int = _compute_internal_pnl(df_test, tau)
        baseline_test_pnl = pnl_test_int.sum()
        res_client = optimal_threshold(df_val, df_test, tau, use_client_specific=True)
        ext_test_pnl = res_client['test_pnl']
        print(f'{tau:>4} | {baseline_test_pnl:>20.2f} | {ext_test_pnl:>20.2f}')
        theta_dict = res_client['theta']
        p_test = task3.predict_adversity(df=df_test, tau=tau)
        for c in sorted(df_test['Name'].unique()):
            mask = (df_test['Name'] == c).values
            th = theta_dict[c]
            pnl_c = np.where(p_test[mask] > th, 0, pnl_test_int[mask]).sum()
            results_rows.append({'client': f'Client{c}', 'tau': tau, 'theta*': round(th, 2), 'final_pnl': round(pnl_c, 2)})
    res_df = pd.DataFrame(results_rows)
    res_df = res_df.sort_values(by=['client', 'tau'])
    here = os.path.dirname(os.path.abspath(__file__))
    res_df.to_csv(os.path.join(here, 'task4_results.csv'), index=False)
    print('=' * 60)
    print('\nSaved task4_results.csv and pnl_vs_theta.png plot.')

if __name__ == '__main__':
    main()