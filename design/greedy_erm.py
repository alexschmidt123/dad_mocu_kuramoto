import numpy as np

def choose_next_pair_greedy(env, candidates):
    g = env.features()
    scores = [env.surrogate.forward_erm(g, xi).item() for xi in candidates]
    return candidates[int(np.argmin(scores))]
