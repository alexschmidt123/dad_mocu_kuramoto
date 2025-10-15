# debug_surrogate_mocu.py
import torch
import yaml
import numpy as np
from surrogate.mpnn_surrogate import MPNNSurrogate
from core.kuramoto_env import PairTestEnv

# Load config and surrogate
cfg = yaml.safe_load(open('configs/config_fast.yaml'))
surrogate = MPNNSurrogate(
    mocu_scale=cfg["surrogate"].get("mocu_scale", 1.0),
    hidden=cfg["surrogate"]["hidden"],
    dropout=cfg["surrogate"]["dropout"]
)
surrogate.load_state_dict(torch.load('models/mpnn_surrogate.pth', map_location='cpu', weights_only=True))
surrogate.eval()

# Create test environment
N, K = cfg["N"], cfg["K"]
rng = np.random.default_rng(42)
omega = rng.uniform(cfg["omega"]["low"], cfg["omega"]["high"], size=N)
prior = (cfg["prior_lower"], cfg["prior_upper"])

env = PairTestEnv(N=N, omega=omega, prior_bounds=prior, K=K, surrogate=surrogate, rng=rng)

# Check MOCU at different stages
print("Initial MOCU (before any experiments):")
initial_mocu = surrogate.forward_mocu(env.features()).item()
print(f"  MOCU: {initial_mocu:.4f}")

# Run random experiments
for step in range(K):
    cands = env.candidate_pairs()
    xi = cands[rng.integers(len(cands))]
    env.step(xi)
    
    mocu = surrogate.forward_mocu(env.features()).item()
    print(f"After experiment {step+1}: MOCU = {mocu:.4f}")

print(f"\nExpected range: 0.1 - 0.2")
print(f"Your range: {initial_mocu:.4f}")