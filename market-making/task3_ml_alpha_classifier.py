from __future__ import annotations
import os
import pickle
import warnings
from functools import lru_cache
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
_TAUS = [5, 10, 15, 20, 25, 30]
_RANDOM_STATE = 42

def _resolve_data_path() -> str:
    env = os.environ.get('TRADE_DATA_CSV')
    if env and os.path.exists(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.path.join(here, 'trade_data.csv'), os.path.join(here, '..', 'ProblemStatement3', 'Question3', 'trade_data.csv')]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError('Cannot find trade_data.csv.')

@lru_cache(maxsize=1)
def _load_raw() -> pd.DataFrame:
    df = pd.read_csv(_resolve_data_path())
    df['_datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['time'].astype(str))
    return df.sort_values('_datetime').reset_index(drop=True)

def _make_label(df: pd.DataFrame, tau: int) -> np.ndarray:
    pnl = df['Side'].astype(float) * (df[f'M{tau}'].astype(float) - df['Trade Price'].astype(float))
    return (pnl < 0).astype(int).to_numpy()

def _split_indices(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_dates = df['Date'].unique()
    train_end = unique_dates[int(len(unique_dates) * 0.6)]
    val_end = unique_dates[int(len(unique_dates) * 0.8)]
    tr_idx = np.where(df['Date'] < train_end)[0]
    va_idx = np.where((df['Date'] >= train_end) & (df['Date'] < val_end))[0]
    te_idx = np.where(df['Date'] >= val_end)[0]
    return (tr_idx, va_idx, te_idx)

def _engineer_features(df: pd.DataFrame, prior_map: Dict[str, float]=None, known_clients: list[str]=None) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    tp = df['Trade Price'].astype(float)
    m0 = df['M0'].astype(float)
    spd = df['Spread'].astype(float)
    vol = df['Volume'].astype(float)
    sd = df['Side'].astype(float)
    name = df['Name'].astype(str)
    dt = df['_datetime'] if '_datetime' in df.columns else pd.to_datetime(df['Date'].astype(str) + ' ' + df['time'].astype(str))
    feats = pd.DataFrame(index=df.index)
    feats['volume_log'] = np.log1p(vol)
    feats['volume_raw'] = vol
    feats['spread_abs'] = spd
    feats['spread_norm'] = spd / (m0 + 1e-10)
    feats['price_vs_mid'] = (tp - m0) / (m0 + 1e-10)
    feats['signed_slip'] = sd * (tp - m0)
    feats['vol_x_spread'] = np.log1p(vol) * spd
    feats['vol_x_slip'] = np.log1p(vol) * feats['signed_slip']
    feats['spread_x_side'] = spd * sd
    open_time = dt.dt.normalize() + pd.Timedelta('09:30:00')
    feats['time_since_open_s'] = (dt - open_time).dt.total_seconds().clip(lower=0)
    feats['hour'] = dt.dt.hour
    feats['minute'] = dt.dt.minute
    feats['dow'] = dt.dt.dayofweek
    if known_clients is None:
        known_clients = sorted(name.unique().tolist())
    if len(known_clients) > 1:
        for c in known_clients[1:]:
            feats[f'cl_{c}'] = (name == c).astype(np.float32)
    if prior_map is None:
        prior_map = {c: 0.5 for c in 'ABCDEF'}
    feats['client_adv_prior'] = name.map(prior_map).fillna(0.5)
    feats['m0_level'] = m0
    m0_past = m0.shift(1)
    spd_past = spd.shift(1)
    vol_past = vol.shift(1)
    sd_past = sd.shift(1)
    for w in (5, 20, 50):
        lag = m0.shift(w)
        feats[f'm0_mom_{w}'] = ((m0 - lag) / (lag.abs() + 1e-10)).fillna(0)
    m0_ret = m0.pct_change().fillna(0)
    for w in (10, 30):
        feats[f'm0_vol_{w}'] = m0_ret.shift(1).rolling(w, min_periods=3).std().fillna(0)
    for w in (10, 30):
        feats[f'spread_roll_{w}'] = spd_past.rolling(w, min_periods=2).mean().fillna(spd)
    spd_mean = spd_past.rolling(30, min_periods=5).mean().fillna(spd)
    spd_std = spd_past.rolling(30, min_periods=5).std().fillna(1.0).replace(0, 1.0)
    feats['spread_zscore'] = ((spd - spd_mean) / spd_std).fillna(0)
    is_buy = (sd_past == 1).astype(float)
    for w in (10, 30):
        feats[f'side_imbal_{w}'] = is_buy.rolling(w, min_periods=3).mean().fillna(0.5)
    dt_unix = (dt.astype(np.int64) // 10 ** 9).astype(float)
    feats['inter_trade_gap_s'] = dt_unix.diff().clip(0, 300).fillna(10)
    feats['avg_gap_30'] = feats['inter_trade_gap_s'].shift(1).rolling(30, min_periods=3).mean().fillna(10)
    vlog = np.log1p(vol_past)
    feats['volume_zscore'] = ((np.log1p(vol) - vlog.rolling(50, min_periods=5).mean().fillna(vlog.mean())) / vlog.rolling(50, min_periods=5).std().fillna(1.0).replace(0, 1.0)).fillna(0)
    day_str = dt.dt.date.astype(str)
    feats['m0_vs_day_open'] = m0.groupby(day_str).transform(lambda x: (x - x.iloc[0]) / (abs(float(x.iloc[0])) + 1e-10)).fillna(0)
    dm0 = m0.diff().abs().shift(1)
    signed_vol = (sd * vol).abs().shift(1)
    kyle_raw = dm0 / (signed_vol + 1e-10)
    feats['kyle_lambda'] = kyle_raw.rolling(20, min_periods=5).mean().fillna(0)
    sv = sd_past * vol_past
    for w in (20, 50):
        feats[f'ofi_{w}'] = sv.rolling(w, min_periods=5).sum().fillna(0)
    total_vol_20 = vol_past.rolling(20, min_periods=5).sum().replace(0, 1)
    feats['ofi_norm'] = (sv.rolling(20, min_periods=5).sum() / total_vol_20).fillna(0)
    return feats.fillna(0).astype(np.float32)
_MODELS: Dict[int, Pipeline] = {}
_PRIOR_MAPS: Dict[int, Dict[str, float]] = {}
_FEAT_NAMES: list[str] = []
_KNOWN_CLIENTS: list[str] = []

def train() -> None:
    global _MODELS, _PRIOR_MAPS, _FEAT_NAMES, _KNOWN_CLIENTS
    df = _load_raw()
    n = len(df)
    tr_idx, va_idx, te_idx = _split_indices(df)
    _KNOWN_CLIENTS = sorted(df['Name'].astype(str).unique().tolist())
    all_metrics = []
    print('Task 3 - Adversity Prediction Model (All Horizons)')
    print(f'Loaded {n:,} trades.')
    for tau in _TAUS:
        print(f'--- Training for tau = {tau} ---')
        y_all = _make_label(df, tau)
        train_df = df.iloc[tr_idx]
        train_y = y_all[tr_idx]
        prior_map = train_df.groupby('Name').apply(lambda x: train_y[x.index].mean()).to_dict()
        _PRIOR_MAPS[tau] = prior_map
        X_df = _engineer_features(df, prior_map, _KNOWN_CLIENTS)
        X_all = X_df.to_numpy(np.float32)
        if not _FEAT_NAMES:
            _FEAT_NAMES = X_df.columns.tolist()
        X_tr, y_tr = (X_all[tr_idx], y_all[tr_idx])
        X_va, y_va = (X_all[va_idx], y_all[va_idx])
        X_te, y_te = (X_all[te_idx], y_all[te_idx])
        best_auc = -1
        best_model = None
        best_C = None
        for C_val in [0.001, 0.01, 0.1, 1.0, 10.0]:
            model = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=C_val, max_iter=1000, solver='lbfgs', random_state=_RANDOM_STATE, n_jobs=-1))])
            model.fit(X_tr, y_tr)
            p_va_temp = model.predict_proba(X_va)[:, 1]
            auc = roc_auc_score(y_va, p_va_temp)
            if auc > best_auc:
                best_auc = auc
                best_model = model
                best_C = C_val
        print(f'  Best C = {best_C} (Val AUC: {best_auc:.4f})')
        _MODELS[tau] = best_model
        p_tr = best_model.predict_proba(X_tr)[:, 1]
        p_va = best_model.predict_proba(X_va)[:, 1]
        p_te = best_model.predict_proba(X_te)[:, 1]
        for tag, y_t, y_p, y_pr in [('train', y_tr, (p_tr >= 0.5).astype(int), p_tr), ('validation', y_va, (p_va >= 0.5).astype(int), p_va), ('test', y_te, (p_te >= 0.5).astype(int), p_te)]:
            all_metrics.append({'tau': tau, 'split': tag, 'accuracy': accuracy_score(y_t, y_p), 'precision': precision_score(y_t, y_p, zero_division=0), 'recall': recall_score(y_t, y_p, zero_division=0), 'log_loss': log_loss(y_t, y_pr)})
    here = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(here, 'task3_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({'models': _MODELS, 'prior_maps': _PRIOR_MAPS, 'feat_names': _FEAT_NAMES, 'known_clients': _KNOWN_CLIENTS}, f)
    print(f'Model saved -> {model_path}')
    metrics_df = pd.DataFrame(all_metrics)
    avg_metrics = metrics_df.groupby('split')[['accuracy', 'precision', 'recall', 'log_loss']].mean().reset_index()
    avg_metrics['sort'] = avg_metrics['split'].map({'train': 0, 'validation': 1, 'test': 2})
    avg_metrics = avg_metrics.sort_values('sort').drop(columns=['sort'])
    results_path = os.path.join(here, 'task3_results.csv')
    avg_metrics.to_csv(results_path, index=False, float_format='%.4f')
    print(f'Results CSV saved -> {results_path}')

def predict_adversity(*args, **kwargs) -> float | np.ndarray:
    if not _MODELS:
        _load_model()
    df = kwargs.get('df')
    if df is None and len(args) > 0:
        df = args[0]
    tau = kwargs.get('tau')
    if tau is None and len(args) > 1:
        tau = args[1]
    if tau not in _TAUS:
        raise ValueError(f'Invalid tau: {tau}. Must be one of {_TAUS}')
    X = _engineer_features(df, _PRIOR_MAPS[tau], _KNOWN_CLIENTS).to_numpy(np.float32)
    preds = _MODELS[tau].predict_proba(X)[:, 1]
    return float(preds[0]) if len(preds) == 1 else preds

def compute_metrics(*args, **kwargs) -> pd.DataFrame:
    df = _load_raw()
    tr_idx, va_idx, te_idx = _split_indices(df)
    if not _MODELS:
        _load_model()
    all_metrics = []
    for tau in _TAUS:
        y_all = _make_label(df, tau)
        X = _engineer_features(df, _PRIOR_MAPS[tau], _KNOWN_CLIENTS).to_numpy(np.float32)
        p_all = _MODELS[tau].predict_proba(X)[:, 1]
        for tag, idx in [('train', tr_idx), ('validation', va_idx), ('test', te_idx)]:
            y_t = y_all[idx]
            y_pr = p_all[idx]
            y_p = (y_pr >= 0.5).astype(int)
            all_metrics.append({'split': tag, 'accuracy': accuracy_score(y_t, y_p), 'precision': precision_score(y_t, y_p, zero_division=0), 'recall': recall_score(y_t, y_p, zero_division=0), 'log_loss': log_loss(y_t, y_pr)})
    metrics_df = pd.DataFrame(all_metrics)
    avg_metrics = metrics_df.groupby('split')[['accuracy', 'precision', 'recall', 'log_loss']].mean().reset_index()
    avg_metrics['sort'] = avg_metrics['split'].map({'train': 0, 'validation': 1, 'test': 2})
    avg_metrics = avg_metrics.sort_values('sort').drop(columns=['sort'])
    avg_metrics.set_index('split', inplace=True)
    return avg_metrics

def _load_model() -> None:
    global _MODELS, _PRIOR_MAPS, _FEAT_NAMES, _KNOWN_CLIENTS
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, 'task3_model.pkl')
    if not os.path.exists(path):
        print(f'{path} not found. Running training to generate model...')
        train()
    with open(path, 'rb') as f:
        art = pickle.load(f)
    _MODELS = art['models']
    _PRIOR_MAPS = art['prior_maps']
    _FEAT_NAMES = art.get('feat_names', [])
    _KNOWN_CLIENTS = art.get('known_clients', ['A', 'B', 'C', 'D', 'E', 'F'])

if __name__ == '__main__':
    train()