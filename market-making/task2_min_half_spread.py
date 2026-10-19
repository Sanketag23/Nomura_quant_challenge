from __future__ import annotations
import os
from functools import lru_cache
from typing import List
import numpy as np
import pandas as pd
_HORIZONS_ALL: tuple[int, ...] = (5, 10, 15, 20, 25, 30)

def _resolve_data_path() -> str:
    env = os.environ.get('TRADE_DATA_CSV')
    if env and os.path.exists(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, 'trade_data.csv'), '/mnt/user-data/uploads/trade_data.csv'):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError('Could not locate trade_data.csv. Set the TRADE_DATA_CSV env-var, or place the CSV next to this script.')

@lru_cache(maxsize=1)
def _load() -> dict:
    df = pd.read_csv(_resolve_data_path())
    names = df['Name'].to_numpy()
    side = df['Side'].to_numpy(np.float64)
    vol = df['Volume'].to_numpy(np.float64)
    tp = df['Trade Price'].to_numpy(np.float64)
    m0 = df['M0'].to_numpy(np.float64)
    M = df[[f'M{t}' for t in _HORIZONS_ALL]].to_numpy(np.float64)
    pnl = side[:, None] * vol[:, None] * (M - tp[:, None])
    pnl_m0 = side[:, None] * vol[:, None] * (M - m0[:, None])
    agg_pnl = pnl.mean(axis=1)
    agg_pnl_m0 = pnl_m0.mean(axis=1)
    return {'names': names, 'pnl': pnl, 'agg_pnl': agg_pnl, 'pnl_m0': pnl_m0, 'agg_pnl_m0': agg_pnl_m0, 'volume': vol, 'tau_index': {t: j for j, t in enumerate(_HORIZONS_ALL)}}

def expected_pnl(client: str, tau: List[int]) -> dict:
    state = _load()
    mask = state['names'] == client
    if not mask.any():
        raise KeyError(f'Unknown client {client!r}; available: {sorted(set(state['names']))}')
    idx = [state['tau_index'][t] for t in tau]
    client_pnl = state['pnl'][mask][:, idx]
    expected_per_horizon = client_pnl.mean(axis=0).tolist()
    expected_agg = float(state['agg_pnl'][mask].mean())
    return {'per_horizon': expected_per_horizon, 'aggregate': expected_agg}

def classify_client(client: str) -> str:
    stats = expected_pnl(client, list(_HORIZONS_ALL))
    return 'profitable' if stats['aggregate'] >= 0 else 'costly'

def min_half_spread(client: str) -> float:
    state = _load()
    mask = state['names'] == client
    if not mask.any():
        raise KeyError(f'Unknown client {client!r}')
    expected_agg_pnl_m0 = float(state['agg_pnl_m0'][mask].mean())
    expected_volume = float(state['volume'][mask].mean())
    if expected_volume == 0:
        return 0.0
    delta_star = -expected_agg_pnl_m0 / expected_volume
    return max(0.0, float(delta_star))

def _build_results_table() -> pd.DataFrame:
    state = _load()
    clients = sorted(set(state['names']))
    rows = []
    for c in clients:
        stats = expected_pnl(c, list(_HORIZONS_ALL))
        delta = min_half_spread(c)
        row = [f'Client{c}'] + stats['per_horizon'] + [stats['aggregate'], delta]
        rows.append(row)
    cols = ['client'] + [f'tau={t}' for t in _HORIZONS_ALL] + ['agg_pnl', 'delta']
    return pd.DataFrame(rows, columns=cols)

if __name__ == '__main__':
    out = _build_results_table()
    target = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'task2_results.csv')
    out.to_csv(target, index=False, float_format='%.4f')
    print(f'Wrote {target}')
    print(out.to_string(index=False))