#!/usr/bin/env python3
"""
Debug script to investigate test environment belief state diversity.
"""

import yaml
import numpy as np
import torch
from surrogate.mpnn_surrogate import MPNNSurrogate
from core.kuramoto_env import PairTestEnv

def load_config(path: str):
    """Load configuration."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def debug_test_environment():
    """Debug test environment belief state diversity."""
    print("="*80)
    print("TEST ENVIRONMENT DEBUG ANALYSIS")
    print("="*80)
    
    # Load config
    cfg = load_config('configs/config_fast.yaml')
    
    # Load surrogate model
    try:
        surrogate = MPNNSurrogate(
            mocu_scale=cfg["surrogate"].get("mocu_scale", 1.0),
            hidden=cfg["surrogate"]["hidden"],
            dropout=cfg["surrogate"]["dropout"]
        )
        state_dict = torch.load('models/mpnn_surrogate.pth', map_location='cpu', weights_only=True)
        surrogate.load_state_dict(state_dict)
        surrogate.eval()
        print("✓ Surrogate model loaded")
    except Exception as e:
        print(f"❌ Failed to load surrogate model: {e}")
        return
    
    # Test multiple episodes
    print(f"\nTesting {50} episodes...")
    mocu_predictions = []
    belief_graphs = []
    
    for episode in range(50):
        # Create environment
        N, K = cfg["N"], cfg["K"]
        rng = np.random.default_rng(cfg["seed"] + episode)
        
        if cfg["omega"]["kind"] == "uniform":
            omega = rng.uniform(cfg["omega"]["low"], cfg["omega"]["high"], size=N)
        else:
            omega = rng.normal(0.0, 1.0, size=N)
        
        prior = (cfg["prior_lower"], cfg["prior_upper"])
        env = PairTestEnv(N=N, omega=omega, prior_bounds=prior, K=K, 
                         surrogate=surrogate, rng=rng)
        
        # Run episode and collect final belief state
        for step in range(K):
            candidates = env.candidate_pairs()
            xi = candidates[np.random.randint(len(candidates))]  # Random choice
            env.step(xi)
        
        # Get final belief graph and MOCU prediction
        final_belief_graph = env.features()
        belief_graphs.append(final_belief_graph)
        
        mocu_pred = surrogate.forward_mocu(final_belief_graph)
        mocu_predictions.append(mocu_pred.item())
        
        if episode < 5:  # Show first 5 episodes
            print(f"  Episode {episode+1}: MOCU = {mocu_predictions[-1]:.6f}")
    
    # Analyze results
    print(f"\nTest Environment MOCU Analysis:")
    print(f"  Episodes: {len(mocu_predictions)}")
    print(f"  Mean: {np.mean(mocu_predictions):.6f}")
    print(f"  Std:  {np.std(mocu_predictions):.6f}")
    print(f"  Min:  {np.min(mocu_predictions):.6f}")
    print(f"  Max:  {np.max(mocu_predictions):.6f}")
    print(f"  Unique values: {len(set(mocu_predictions))}")
    
    if np.std(mocu_predictions) < 1e-6:
        print("⚠️  WARNING: Test environment produces identical MOCU predictions!")
        print("   This explains the zero variance in test results.")
    elif np.std(mocu_predictions) < 0.01:
        print("⚠️  WARNING: Test environment produces very similar MOCU predictions!")
        print("   This explains the near-zero variance in test results.")
    else:
        print("✓ Test environment produces diverse MOCU predictions")
    
    # Debug belief graph diversity
    print(f"\nBelief Graph Analysis:")
    if belief_graphs:
        # Check node features diversity
        node_feats = [bg['node_feats'] for bg in belief_graphs]
        node_feats_array = np.array(node_feats)
        
        print(f"  Node features shape: {node_feats_array.shape}")
        print(f"  Node features std: {np.std(node_feats_array):.6f}")
        
        # Check edge features diversity
        edge_feats = [bg['edge_feats'] for bg in belief_graphs if len(bg['edge_feats']) > 0]
        if edge_feats:
            edge_feats_array = np.concatenate(edge_feats, axis=0)
            print(f"  Edge features shape: {edge_feats_array.shape}")
            print(f"  Edge features std: {np.std(edge_feats_array):.6f}")
        
        # Check if belief graphs are too similar
        if len(belief_graphs) > 1:
            # Compare first two belief graphs
            bg1, bg2 = belief_graphs[0], belief_graphs[1]
            
            node_diff = np.mean(np.abs(bg1['node_feats'] - bg2['node_feats']))
            print(f"  Node features difference (first 2): {node_diff:.6f}")
            
            if len(bg1['edge_feats']) > 0 and len(bg2['edge_feats']) > 0:
                edge_diff = np.mean(np.abs(bg1['edge_feats'] - bg2['edge_feats']))
                print(f"  Edge features difference (first 2): {edge_diff:.6f}")

def debug_greedy_vs_random():
    """Compare greedy MPNN vs random selection in test environment."""
    print("\n" + "="*80)
    print("GREEDY MPNN vs RANDOM COMPARISON")
    print("="*80)
    
    cfg = load_config('configs/config_fast.yaml')
    
    # Load surrogate
    surrogate = MPNNSurrogate(
        mocu_scale=cfg["surrogate"].get("mocu_scale", 1.0),
        hidden=cfg["surrogate"]["hidden"],
        dropout=cfg["surrogate"]["dropout"]
    )
    state_dict = torch.load('models/mpnn_surrogate.pth', map_location='cpu', weights_only=True)
    surrogate.load_state_dict(state_dict)
    surrogate.eval()
    
    # Test both strategies
    strategies = {
        'Random': lambda env, cands: cands[np.random.randint(len(cands))],
        'Greedy': lambda env, cands: choose_next_pair_greedy(env, cands)
    }
    
    from design.greedy_erm import choose_next_pair_greedy
    
    for strategy_name, chooser_fn in strategies.items():
        print(f"\nTesting {strategy_name} strategy:")
        mocu_predictions = []
        
        for episode in range(10):  # Test 10 episodes
            # Create environment
            N, K = cfg["N"], cfg["K"]
            rng = np.random.default_rng(42 + episode)  # Fixed seed for comparison
            
            if cfg["omega"]["kind"] == "uniform":
                omega = rng.uniform(cfg["omega"]["low"], cfg["omega"]["high"], size=N)
            else:
                omega = rng.normal(0.0, 1.0, size=N)
            
            prior = (cfg["prior_lower"], cfg["prior_upper"])
            env = PairTestEnv(N=N, omega=omega, prior_bounds=prior, K=K, 
                             surrogate=surrogate, rng=rng)
            
            # Run episode
            for step in range(K):
                candidates = env.candidate_pairs()
                xi = chooser_fn(env, candidates)
                env.step(xi)
            
            # Get final MOCU prediction
            final_belief_graph = env.features()
            mocu_pred = surrogate.forward_mocu(final_belief_graph)
            mocu_predictions.append(mocu_pred.item())
        
        print(f"  MOCU predictions: {[f'{x:.6f}' for x in mocu_predictions[:5]]}...")
        print(f"  Mean: {np.mean(mocu_predictions):.6f}")
        print(f"  Std:  {np.std(mocu_predictions):.6f}")
        print(f"  Unique: {len(set(mocu_predictions))}")

def main():
    debug_test_environment()
    debug_greedy_vs_random()
    
    print("\n" + "="*80)
    print("DEBUG ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
