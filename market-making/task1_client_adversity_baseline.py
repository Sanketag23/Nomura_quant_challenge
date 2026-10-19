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
    raise FileNotFoundError('Could not locate trade_data.csv.  Set the TRADE_DATA_CSV env-var, or place the CSV next to this script.')

@lru_cache(maxsize=1)
def _load() -> dict:
    df = pd.read_csv(_resolve_data_path())
    side = df['Side'].to_numpy(np.float64)
    tp = df['Trade Price'].to_numpy(np.float64)
    M = df[[f'M{t}' for t in _HORIZONS_ALL]].to_numpy(np.float64)
    signed_drift = side[:, None] * (M - tp[:, None])
    adverse = signed_drift < 0
    return {'names': df['Name'].to_numpy(), 'adverse': adverse, 'tau_index': {t: j for j, t in enumerate(_HORIZONS_ALL)}}

def adversity_profile(client: str, tau: List[int]) -> List[float]:
    state = _load()
    mask = state['names'] == client
    if not mask.any():
        raise KeyError(f'Unknown client {client!r}; available: {sorted(set(state['names']))}')
    adv = state['adverse'][mask]
    idx = [state['tau_index'][t] for t in tau]
    return (adv[:, idx].mean(axis=0) * 100.0).tolist()

def _build_results_table() -> pd.DataFrame:
    state = _load()
    clients = sorted(set(state['names']))
    rows = []
    for c in clients:
        rows.append([f'Client{c}'] + adversity_profile(c, list(_HORIZONS_ALL)))
    cols = ['client'] + [f'tau={t}' for t in _HORIZONS_ALL]
    return pd.DataFrame(rows, columns=cols)

if __name__ == '__main__':
    out = _build_results_table()
    target = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'task1_results.csv')
    out.to_csv(target, index=False, float_format='%.4f')
    print(f'Wrote {target}')
    print(out.to_string(index=False))