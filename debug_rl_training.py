#!/usr/bin/env python3
"""
Debug script to diagnose behavior cloning training issues.
"""

import yaml
import numpy as np
import torch
from surrogate.mpnn_surrogate import MPNNSurrogate
from design.dad_policy import DADPolicy
from design.greedy_erm import choose_next_pair_greedy
from design.train_rl import make_hist_tokens
from core.kuramoto_env import PairTestEnv

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def make_env_factory(cfg, surrogate):
    def env_factory():
        N, K = cfg["N"], cfg["K"]
        rng = np.random.default_rng(cfg["seed"])
        
        if cfg["omega"]["kind"] == "uniform":
            omega = rng.uniform(cfg["omega"]["low"], cfg["omega"]["high"], size=N)
        else:
            omega = rng.normal(0.0, 1.0, size=N)
        
        prior = (cfg["prior_lower"], cfg["prior_upper"])
        return PairTestEnv(N=N, omega=omega, prior_bounds=prior, K=K, 
                          surrogate=surrogate, rng=rng)
    return env_factory

def debug_single_step():
    """Debug a single training step in detail."""
    print("="*80)
    print("DEBUGGING SINGLE RL TRAINING STEP")
    print("="*80)
    
    cfg = load_config('configs/config_fast.yaml')
    
    # Load surrogate
    surrogate = MPNNSurrogate(
        mocu_scale=cfg["surrogate"].get("mocu_scale", 1.0),
        hidden=cfg["surrogate"]["hidden"],
        dropout=cfg["surrogate"]["dropout"]
    )
    surrogate.load_state_dict(torch.load('models/mpnn_surrogate.pth', map_location='cpu', weights_only=True))
    surrogate.eval()
    
    # Create environment
    env_factory = make_env_factory(cfg, surrogate)
    env = env_factory()
    
    # Initialize policy (untrained)
    policy = DADPolicy(hidden=cfg["dad_rl"]["hidden"])
    
    print("\n1. Environment setup:")
    print(f"   N = {env.N}, K = {env.K}")
    print(f"   Omega = {env.omega}")
    
    # Get candidates
    cands = env.candidate_pairs()
    print(f"\n2. Candidate pairs: {cands}")
    
    # Get expert choice
    xi_star = choose_next_pair_greedy(env, cands)
    expert_idx = cands.index(xi_star)
    print(f"\n3. Expert (Greedy) choice: {xi_star} (index {expert_idx})")
    
    # Get policy scores (before training)
    hist_tokens = make_hist_tokens(env.h, env.N)
    g = env.features()
    
    print(f"\n4. Belief graph features:")
    print(f"   Node features shape: {g['node_feats'].shape}")
    print(f"   Edge features shape: {g['edge_feats'].shape}")
    print(f"   History tokens: {hist_tokens}")
    
    with torch.no_grad():
        scores = policy.forward(g, cands, hist_tokens, env.N)
    
    print(f"\n5. Policy scores (untrained):")
    for i, (cand, score) in enumerate(zip(cands, scores)):
        marker = " ← EXPERT" if i == expert_idx else ""
        print(f"   {cand}: {score.item():.4f}{marker}")
    
    pred_idx = scores.argmin().item()
    print(f"\n6. Policy prediction: {cands[pred_idx]} (index {pred_idx})")
    print(f"   Correct: {pred_idx == expert_idx}")
    
    # Compute loss (hinge)
    expert_score = scores[expert_idx]
    margin = 0.5
    loss = 0.0
    for i in range(len(scores)):
        if i != expert_idx:
            loss += torch.relu(expert_score - scores[i] + margin)
    loss = loss / (len(cands) - 1)
    
    print(f"\n7. Hinge Loss: {loss.item():.4f}")
    
    # Check if loss makes sense
    print(f"\n8. Loss breakdown:")
    for i in range(len(scores)):
        if i != expert_idx:
            term = expert_score - scores[i] + margin
            print(f"   Expert vs {cands[i]}: {expert_score.item():.4f} - {scores[i].item():.4f} + 0.5 = {term.item():.4f}")
            print(f"     → relu = {torch.relu(term).item():.4f}")
    
    # Also compute CrossEntropy loss
    logits = -scores.unsqueeze(0)
    target = torch.tensor([expert_idx], dtype=torch.long)
    ce_loss = torch.nn.functional.cross_entropy(logits, target)
    print(f"\n9. CrossEntropy Loss: {ce_loss.item():.4f}")

def debug_score_distribution():
    """Check if policy produces varied scores."""
    print("\n" + "="*80)
    print("CHECKING SCORE DISTRIBUTION")
    print("="*80)
    
    cfg = load_config('configs/config_fast.yaml')
    
    # Load surrogate
    surrogate = MPNNSurrogate(
        mocu_scale=cfg["surrogate"].get("mocu_scale", 1.0),
        hidden=cfg["surrogate"]["hidden"],
        dropout=cfg["surrogate"]["dropout"]
    )
    surrogate.load_state_dict(torch.load('models/mpnn_surrogate.pth', map_location='cpu', weights_only=True))
    surrogate.eval()
    
    # Try both untrained and trained policy
    for policy_name, policy_path in [("Untrained", None), ("Trained", 'models/dad_policy.pth')]:
        print(f"\n{policy_name} Policy:")
        
        policy = DADPolicy(hidden=cfg["dad_rl"]["hidden"])
        if policy_path:
            try:
                policy.load_state_dict(torch.load(policy_path, map_location='cpu'))
                policy.eval()
            except Exception as e:
                print(f"  (Model not found or error: {e})")
                continue
        
        env_factory = make_env_factory(cfg, surrogate)
        
        all_scores = []
        all_predictions = []
        
        for trial in range(5):
            env = env_factory()
            cands = env.candidate_pairs()
            hist_tokens = []
            g = env.features()
            
            with torch.no_grad():
                scores = policy.forward(g, cands, hist_tokens, env.N)
            
            all_scores.extend(scores.tolist())
            pred_idx = scores.argmin().item()
            all_predictions.append(cands[pred_idx])
        
        print(f"  Score statistics over 5 environments:")
        print(f"    Mean: {np.mean(all_scores):.4f}")
        print(f"    Std: {np.std(all_scores):.4f}")
        print(f"    Min: {np.min(all_scores):.4f}")
        print(f"    Max: {np.max(all_scores):.4f}")
        print(f"  Unique predictions: {len(set(all_predictions))}/5")
        print(f"  Predictions: {all_predictions}")

def debug_greedy_consistency():
    """Check if Greedy makes consistent choices."""
    print("\n" + "="*80)
    print("CHECKING GREEDY CONSISTENCY")
    print("="*80)
    
    cfg = load_config('configs/config_fast.yaml')
    
    # Load surrogate
    surrogate = MPNNSurrogate(
        mocu_scale=cfg["surrogate"].get("mocu_scale", 1.0),
        hidden=cfg["surrogate"]["hidden"],
        dropout=cfg["surrogate"]["dropout"]
    )
    surrogate.load_state_dict(torch.load('models/mpnn_surrogate.pth', map_location='cpu', weights_only=True))
    surrogate.eval()
    
    env_factory = make_env_factory(cfg, surrogate)
    
    greedy_choices = []
    for trial in range(10):
        env = env_factory()
        cands = env.candidate_pairs()
        xi_star = choose_next_pair_greedy(env, cands)
        greedy_choices.append(xi_star)
    
    print(f"\nGreedy choices over 10 trials (same initial state):")
    print(f"  Choices: {greedy_choices}")
    print(f"  Unique: {len(set(greedy_choices))}/10")
    
    if len(set(greedy_choices)) == 1:
        print("  ✓ Greedy is deterministic (good)")
    else:
        print("  ⚠️  WARNING: Greedy is non-deterministic!")

def main():
    print("="*80)
    print("BEHAVIOR CLONING DEBUG DIAGNOSTICS")
    print("="*80)
    
    try:
        debug_single_step()
        debug_score_distribution()
        debug_greedy_consistency()
        
        print("\n" + "="*80)
        print("DIAGNOSTICS COMPLETE")
        print("="*80)
        
    except Exception as e:
        print(f"\nError during diagnostics: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()