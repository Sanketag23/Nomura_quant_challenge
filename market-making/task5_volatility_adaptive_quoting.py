import os
import numpy as np
import pandas as pd
import task3
C_SIGMA = 1.0
C_ALPHA = 4.0
C_ETA = 5.0
_ALPHA_BASELINE = 0.487

def quote(inventory: float, sigma: float, alpha: float, eta: float) -> tuple[float, float]:
    delta_base = C_SIGMA * sigma
    global _ALPHA_BASELINE
    ALPHA_BASELINE = _ALPHA_BASELINE
    toxicity_multiplier = 1.0 + C_ALPHA * max(0.0, alpha - ALPHA_BASELINE)
    delta_spread = delta_base * toxicity_multiplier
    urgency = 0.1 + C_ETA * eta ** 3
    INV_NORMALIZER = 500.0
    skew = inventory / INV_NORMALIZER * sigma * urgency
    delta_bid = delta_spread + skew
    delta_ask = delta_spread - skew
    c_min = 0.5
    delta_max = 50.0
    delta_bid = max(c_min * sigma, min(delta_bid, delta_max))
    delta_ask = max(c_min * sigma, min(delta_ask, delta_max))
    return (float(delta_bid), float(delta_ask))

def validate_quote(df_test: pd.DataFrame, alpha_preds: np.ndarray, lambda_: float=0.8, gamma: float=1.0, phi: float=0.0001, seed: int=42) -> dict:
    inventory = 0.0
    daily_pnls = []
    current_day = None
    daily_pnl = 0.0
    day_sigma = []
    df_test = df_test.reset_index(drop=True)
    np.random.seed(seed)
    random_draws = np.random.uniform(0, 1, len(df_test))
    alpha_baselines = pd.Series(alpha_preds).expanding().mean().shift(1).fillna(0.487).values
    dt = pd.to_datetime(df_test['Date'].astype(str) + ' ' + df_test['time'].astype(str))
    open_time = dt.dt.normalize() + pd.Timedelta('09:30:00')
    elapsed_s = (dt - open_time).dt.total_seconds().clip(lower=0)
    total_day_s = 6.5 * 3600
    etas = (elapsed_s / total_day_s).clip(upper=1.0).values
    m0_arr = df_test['M0'].values
    side_arr = df_test['Side'].values
    vol_arr = df_test['Volume'].values
    date_arr = df_test['Date'].values
    taus = [5, 10, 15, 20, 25, 30]
    M_taus = df_test[[f'M{t}' for t in taus]].values
    M_tau_mean = M_taus.mean(axis=1)
    returns = df_test['M0'].pct_change().fillna(0).values
    r2 = returns ** 2
    rolling_r2 = pd.Series(r2).shift(1).rolling(20, min_periods=2).mean().fillna(1e-08).values
    sigmas = np.sqrt(rolling_r2)
    sigmas_bps = sigmas * 10000.0
    total_trades = 0
    total_fills = 0
    total_penalty = 0.0
    cumulative_pnl = 0.0
    peak_pnl = 0.0
    max_drawdown = 0.0
    for i in range(len(df_test)):
        date = date_arr[i]
        if date != current_day:
            if current_day is not None:
                avg_sigma_day = np.mean(day_sigma) if day_sigma else 1.0
                penalty = phi * inventory ** 2 * avg_sigma_day
                daily_pnl -= penalty
                daily_pnls.append(daily_pnl)
                total_penalty += penalty
                cumulative_pnl -= penalty
                if cumulative_pnl > peak_pnl:
                    peak_pnl = cumulative_pnl
                drawdown = peak_pnl - cumulative_pnl
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                inventory = 0.0
            current_day = date
            daily_pnl = 0.0
            day_sigma = []
        sigma_t = sigmas_bps[i]
        alpha_t = alpha_preds[i]
        eta_t = etas[i]
        day_sigma.append(sigma_t)
        global _ALPHA_BASELINE
        _ALPHA_BASELINE = alpha_baselines[i]
        db, da = quote(inventory, sigma_t, alpha_t, eta_t)
        actual_side = side_arr[i]
        v = vol_arr[i]
        delta_quoted = db if actual_side == 1 else da
        p_fill = lambda_ * np.exp(-gamma * delta_quoted / (sigma_t + 1e-08))
        is_filled = random_draws[i] < p_fill
        total_trades += 1
        if is_filled:
            total_fills += 1
            inventory += actual_side * v
            price_delta = m0_arr[i] * (delta_quoted / 10000.0)
            trade_pnl = v * price_delta + actual_side * v * (M_tau_mean[i] - m0_arr[i])
            daily_pnl += trade_pnl
            cumulative_pnl += trade_pnl
            if cumulative_pnl > peak_pnl:
                peak_pnl = cumulative_pnl
            drawdown = peak_pnl - cumulative_pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown
    if current_day is not None:
        avg_sigma_day = np.mean(day_sigma) if day_sigma else 1.0
        penalty = phi * inventory ** 2 * avg_sigma_day
        daily_pnl -= penalty
        daily_pnls.append(daily_pnl)
        total_penalty += penalty
        cumulative_pnl -= penalty
        if cumulative_pnl > peak_pnl:
            peak_pnl = cumulative_pnl
        drawdown = peak_pnl - cumulative_pnl
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    total_pnl = sum(daily_pnls)
    std_pnl = np.std(daily_pnls) if len(daily_pnls) > 1 else 1.0
    sigma_floor = 100.0
    score = total_pnl / max(std_pnl, sigma_floor)
    return {'Total Fills': total_fills, 'Fill Rate (%)': total_fills / total_trades * 100, 'Total EOD Penalties': total_penalty, 'Total Net PnL': total_pnl, 'Daily PnL StdDev': std_pnl, 'Max Drawdown': max_drawdown, 'Sharpe Score': score}

def tune_parameters(df_val: pd.DataFrame, alpha_preds: np.ndarray):
    global C_SIGMA, C_ALPHA, C_ETA
    best_sharpe = -np.inf
    best_params = {}
    print('\n--- Tuning Hyperparameters ---')
    for c_sig in [0.5, 1.0, 1.5]:
        for c_alp in [0.0, 2.0, 4.0, 6.0, 8.0]:
            for c_et in [1.0, 5.0, 10.0, 20.0]:
                C_SIGMA = c_sig
                C_ALPHA = c_alp
                C_ETA = c_et
                res = validate_quote(df_val, alpha_preds)
                sharpe = res['Sharpe Score']
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = {'C_SIGMA': c_sig, 'C_ALPHA': c_alp, 'C_ETA': c_et}
                    print(f'New Best! Sharpe: {sharpe:5.2f} | PnL: {res['Total Net PnL']:8.2f} | Fills: {res['Total Fills']:5} | Params: {best_params}')
    print(f'\nOptimal Parameters: {best_params} (Sharpe: {best_sharpe:.2f})')
    C_SIGMA = best_params['C_SIGMA']
    C_ALPHA = best_params['C_ALPHA']
    C_ETA = best_params['C_ETA']
    return best_params

def main():
    global C_SIGMA, C_ALPHA, C_ETA
    print('Loading data and model...')
    df = task3._load_raw()
    tr_idx, va_idx, te_idx = task3._split_indices(df)
    df_val = df.iloc[va_idx].copy()
    df_test = df.iloc[te_idx].copy()
    print('Generating alpha scores (tau=30)...')
    alpha_preds_val = task3.predict_adversity(df=df_val, tau=30)
    alpha_preds_test = task3.predict_adversity(df=df_test, tau=30)
    tune_parameters(df_val, alpha_preds_val)
    print('\n--- Running Final Simulator (Validation Set) ---')
    res_val = validate_quote(df_val, alpha_preds_val)
    for k, v in res_val.items():
        print(f'  {k}: {v:,.2f}')
    print('\n--- Running Final Simulator (Held-out Test Set | Robust Seed Average) ---')
    seeds = [42, 1, 2, 7, 100]
    test_sharpes = []
    for seed in seeds:
        res_test = validate_quote(df_test, alpha_preds_test, seed=seed)
        test_sharpes.append(res_test['Sharpe Score'])
    avg_sharpe = np.mean(test_sharpes)
    std_sharpe = np.std(test_sharpes)
    print(f'  Test Sharpe Score: {avg_sharpe:.2f} ± {std_sharpe:.2f} (over {len(seeds)} seeds)')
    print('  Representative Run Metrics:')
    for k, v in res_test.items():
        if k != 'Sharpe Score':
            print(f'    {k}: {v:,.2f}')
    print('\n--- Running Final Simulator (Entire Dataset: Train+Val+Test) ---')
    alpha_preds_all = task3.predict_adversity(df=df, tau=30)
    res_all = validate_quote(df, alpha_preds_all)
    for k, v in res_all.items():
        if k != 'Sharpe Score':
            print(f'  {k}: {v:,.2f}')
        else:
            print(f'  {k}: {v:.2f}')

if __name__ == '__main__':
    main()