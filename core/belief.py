from dataclasses import dataclass
import numpy as np

@dataclass
class History:
    pairs: list          # [(i,j), ...]
    outcomes: list       # [1(sync)/0(not), ...]
    lower: np.ndarray    # (N,N) symmetric lower bounds
    upper: np.ndarray    # (N,N) symmetric upper bounds
    tested: np.ndarray   # (N,N) bool mask for tested edges

def init_history(N: int, prior_bounds) -> History:
    """
    Initialize history with either:
    - Tuple (lo, hi): uniform bounds
    - Tuple (lower_matrix, upper_matrix): adaptive bounds
    """
    lower = np.zeros((N, N))
    upper = np.zeros((N, N))
    tested = np.zeros((N, N), dtype=bool)
    
    if isinstance(prior_bounds, tuple) and len(prior_bounds) == 2:
        if isinstance(prior_bounds[0], (int, float)):
            # Uniform bounds
            lo, hi = prior_bounds
            for i in range(N):
                for j in range(N):
                    if i == j: continue
                    lower[i, j] = lo
                    upper[i, j] = hi
        else:
            # Matrix bounds (adaptive)
            lower = prior_bounds[0].copy()
            upper = prior_bounds[1].copy()
    
    return History(pairs=[], outcomes=[], lower=lower, upper=upper, tested=tested)

def pair_threshold(omega: np.ndarray, i: int, j: int) -> float:
    return 0.5 * abs(omega[i] - omega[j])

def update_intervals(h: History, xi: tuple[int,int], y_sync: bool, omega: np.ndarray):
    i, j = xi
    lam = pair_threshold(omega, i, j)
    if y_sync:
        h.lower[i, j] = max(h.lower[i, j], lam); h.lower[j, i] = h.lower[i, j]
    else:
        h.upper[i, j] = min(h.upper[i, j], lam); h.upper[j, i] = h.upper[i, j]
    h.tested[i, j] = h.tested[j, i] = True
    h.pairs.append((i, j)); h.outcomes.append(1 if y_sync else 0)

def build_belief_graph(h: History, omega: np.ndarray) -> dict:
    N = omega.shape[0]
    node_feats = np.stack([omega, np.ones_like(omega)], axis=1)  # (N,2)
    edges, e_feats = [], []
    for i in range(N):
        for j in range(i+1, N):
            lam = pair_threshold(omega, i, j)
            lo, up = h.lower[i, j], h.upper[i, j]
            wid = max(0.0, up - lo)
            tfl = 1.0 if h.tested[i, j] else 0.0
            edges.append((i, j)); e_feats.append([lo, up, wid, lam, tfl])
    edge_index = np.array(edges, dtype=int).T if edges else np.zeros((2,0), dtype=int)
    edge_feats = np.array(e_feats, dtype=float) if edges else np.zeros((0,5), dtype=float)
    return dict(node_feats=node_feats, edge_index=edge_index, edge_feats=edge_feats)
