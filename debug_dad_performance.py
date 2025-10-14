#!/usr/bin/env python3
"""
Debug script to analyze why DAD is performing poorly.
"""

import yaml
import numpy as np
import torch
from surrogate.mpnn_surrogate import MPNNSurrogate
from design.dad_policy import DADPolicy
from design.greedy_erm import choose_next_pair_greedy
from core.kuramoto_env import PairTestEnv
from design.train_bc import make_hist_tokens

def load_config(path: str):
    """Load configuration."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def debug_dad_vs_greedy_decisions():
    """Compare DAD vs Greedy decisions on the same environments."""
    print("="*80)
    print("DAD vs GREEDY DECISION COMPARISON")
    print("="*80)
    
    cfg = load_config('configs/config_fast.yaml')
    
    # Load models
    surrogate = MPNNSurrogate(
        mocu_scale=cfg["surrogate"].get("mocu_scale", 1.0),
        hidden=cfg["surrogate"]["hidden"],
        dropout=cfg["surrogate"]["dropout"]
    )
    state_dict = torch.load('models/mpnn_surrogate.pth', map_location='cpu', weights_only=True)
    surrogate.load_state_dict(state_dict)
    surrogate.eval()
    
    policy = DADPolicy(hidden=cfg["dad_bc"]["hidden"])
    policy.load_state_dict(torch.load('models/dad_policy.pth', map_location='cpu'))
    policy.eval()
    
    print("✓ Models loaded successfully")
    
    # Test on 10 identical environments
    decisions_match = 0
    dad_better_mocu = 0
    greedy_better_mocu = 0
    
    print(f"\nTesting on 10 identical environments:")
    
    for episode in range(10):
        # Create identical environment
        N, K = cfg["N"], cfg["K"]
        rng = np.random.default_rng(42 + episode)
        
        if cfg["omega"]["kind"] == "uniform":
            omega = rng.uniform(cfg["omega"]["low"], cfg["omega"]["high"], size=N)
        else:
            omega = rng.normal(0.0, 1.0, size=N)
        
        prior = (cfg["prior_lower"], cfg["prior_upper"])
        
        # Create two identical environments
        env_greedy = PairTestEnv(N=N, omega=omega, prior_bounds=prior, K=K, 
                               surrogate=surrogate, rng=rng)
        env_dad = PairTestEnv(N=N, omega=omega, prior_bounds=prior, K=K, 
                             surrogate=surrogate, rng=rng)
        
        # Run both strategies
        greedy_decisions = []
        dad_decisions = []
        
        for step in range(K):
            candidates = env_greedy.candidate_pairs()
            
            # Greedy decision
            greedy_choice = choose_next_pair_greedy(env_greedy, candidates)
            greedy_decisions.append(greedy_choice)
            env_greedy.step(greedy_choice)
            
            # DAD decision
            hist_tokens = make_hist_tokens(env_dad.h, env_dad.N)
            dad_choice = policy.choose(env_dad, hist_tokens)
            dad_decisions.append(dad_choice)
            env_dad.step(dad_choice)
        
        # Compare decisions
        decisions_identical = greedy_decisions == dad_decisions
        if decisions_identical:
            decisions_match += 1
        
        # Compare final MOCU
        final_mocu_greedy = surrogate.forward_mocu(env_greedy.features()).item()
        final_mocu_dad = surrogate.forward_mocu(env_dad.features()).item()
        
        if final_mocu_dad < final_mocu_greedy:
            dad_better_mocu += 1
        elif final_mocu_greedy < final_mocu_dad:
            greedy_better_mocu += 1
        
        print(f"  Episode {episode+1}:")
        print(f"    Greedy decisions: {greedy_decisions}")
        print(f"    DAD decisions:    {dad_decisions}")
        print(f"    Identical: {decisions_identical}")
        print(f"    Greedy MOCU: {final_mocu_greedy:.6f}")
        print(f"    DAD MOCU:    {final_mocu_dad:.6f}")
        print(f"    DAD better:  {final_mocu_dad < final_mocu_greedy}")
        print()
    
    print(f"Summary:")
    print(f"  Identical decisions: {decisions_match}/10 ({100*decisions_match/10:.1f}%)")
    print(f"  DAD better MOCU: {dad_better_mocu}/10 ({100*dad_better_mocu/10:.1f}%)")
    print(f"  Greedy better MOCU: {greedy_better_mocu}/10 ({100*greedy_better_mocu/10:.1f}%)")
    
    if decisions_match == 10:
        print("⚠️  WARNING: DAD always makes identical decisions to Greedy!")
        print("   This suggests DAD is just copying Greedy behavior.")
    elif decisions_match > 7:
        print("⚠️  WARNING: DAD makes very similar decisions to Greedy!")
        print("   DAD might not be learning to improve beyond Greedy.")

def debug_dad_policy_learning():
    """Debug DAD policy learning process."""
    print("\n" + "="*80)
    print("DAD POLICY LEARNING ANALYSIS")
    print("="*80)
    
    cfg = load_config('configs/config_fast.yaml')
    
    # Load DAD policy
    policy = DADPolicy(hidden=cfg["dad_bc"]["hidden"])
    policy.load_state_dict(torch.load('models/dad_policy.pth', map_location='cpu'))
    policy.eval()
    
    # Test policy on different belief states
    print("Testing DAD policy on different belief states:")
    
    # Create test environment
    N, K = cfg["N"], cfg["K"]
    rng = np.random.default_rng(42)
    
    if cfg["omega"]["kind"] == "uniform":
        omega = rng.uniform(cfg["omega"]["low"], cfg["omega"]["high"], size=N)
    else:
        omega = rng.normal(0.0, 1.0, size=N)
    
    prior = (cfg["prior_lower"], cfg["prior_upper"])
    env = PairTestEnv(N=N, omega=omega, prior_bounds=prior, K=K, 
                     surrogate=None, rng=rng)
    
    # Test policy at different steps
    for step in range(K):
        candidates = env.candidate_pairs()
        hist_tokens = make_hist_tokens(env.h, env.N)
        
        # Get policy scores for all candidates
        scores = policy.forward(env.features(), candidates, hist_tokens, env.N)
        scores_np = scores.detach().cpu().numpy()
        
        # Choose best candidate
        best_idx = np.argmin(scores_np)
        chosen_pair = candidates[best_idx]
        
        print(f"  Step {step+1}:")
        print(f"    Available pairs: {candidates}")
        print(f"    Policy scores: {[f'{s:.4f}' for s in scores_np]}")
        print(f"    Chosen pair: {chosen_pair}")
        print(f"    Score variance: {np.var(scores_np):.6f}")
        
        # Check if policy has preferences
        if np.var(scores_np) < 1e-6:
            print("    ⚠️  WARNING: Policy scores are nearly identical!")
            print("       Policy might not be learning to distinguish candidates.")
        
        # Make random choice and continue
        random_choice = candidates[np.random.randint(len(candidates))]
        env.step(random_choice)

def debug_training_data_quality():
    """Debug the quality of training data for DAD."""
    print("\n" + "="*80)
    print("DAD TRAINING DATA QUALITY ANALYSIS")
    print("="*80)
    
    # This would require checking the behavior cloning training process
    print("DAD is trained via behavior cloning on Greedy MPNN trajectories.")
    print("If Greedy MPNN is not optimal, DAD learns suboptimal behavior.")
    print()
    print("To improve DAD performance:")
    print("1. Use a better expert policy (e.g., optimal policy if available)")
    print("2. Increase training episodes for DAD")
    print("3. Use reinforcement learning instead of behavior cloning")
    print("4. Add exploration during DAD training")

def main():
    debug_dad_vs_greedy_decisions()
    debug_dad_policy_learning()
    debug_training_data_quality()
    
    print("\n" + "="*80)
    print("DAD PERFORMANCE DEBUG COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
