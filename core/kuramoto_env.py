import numpy as np
from core.belief import init_history, update_intervals, pair_threshold, build_belief_graph

class PairTestEnv:
    def __init__(self, N, omega, prior_bounds, K, surrogate=None, rng=None):
        self.N, self.omega, self.K = N, omega.astype(float), K
        self.surrogate = surrogate
        self.rng = np.random.default_rng() if rng is None else rng
        self.h = init_history(N, prior_bounds)
        self.A_true = self.sample_true_A(prior_bounds)

    def sample_true_A(self, prior_bounds):
        lo, hi = prior_bounds
        A = self.rng.uniform(lo, hi, size=(self.N, self.N))
        A = 0.5*(A + A.T); np.fill_diagonal(A, 0.0)
        return A

    def reset(self):
        self.h.pairs.clear(); self.h.outcomes.clear()

    def step(self, xi: tuple[int,int]) -> dict:
        i, j = xi
        lam = pair_threshold(self.omega, i, j)
        y_sync = (self.A_true[i, j] >= lam)
        update_intervals(self.h, xi, y_sync, self.omega)
        return {"y": y_sync, "h": self.h}

    def features(self) -> dict:
        return build_belief_graph(self.h, self.omega)

    def candidate_pairs(self):
        return [(i,j) for i in range(self.N) for j in range(i+1,self.N)]
