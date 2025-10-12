#!/usr/bin/env python3
"""
Simple test script to verify the demo works correctly.
"""

import sys
import os
import numpy as np
import torch

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from core.kuramoto_env import PairTestEnv
        from core.belief import init_history, update_intervals, pair_threshold
        from core.pacemaker_control import simulate_with_pacemaker, sync_check
        from core.bisection import find_min_a_ctrl
        from surrogate.mpnn_surrogate import MPNNSurrogate
        from design.greedy_erm import choose_next_pair_greedy
        from design.dad_policy import DADPolicy
        from eval.run_eval import run_episode
        from data_generation.synthetic_data import SyntheticDataGenerator
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of core components."""
    print("\nTesting basic functionality...")
    
    try:
        from core.kuramoto_env import PairTestEnv
        from surrogate.mpnn_surrogate import MPNNSurrogate
        
        # Test environment creation
        N, K = 5, 4
        omega = np.random.uniform(-1.0, 1.0, size=N)
        prior_bounds = (0.05, 0.50)
        
        env = PairTestEnv(N=N, omega=omega, prior_bounds=prior_bounds, K=K)
        print("✓ Environment creation successful")
        
        # Test belief operations
        belief_graph = env.features()
        print("✓ Belief graph creation successful")
        
        # Test pair selection
        candidates = env.candidate_pairs()
        print(f"✓ Found {len(candidates)} candidate pairs")
        
        # Test experiment step
        xi = candidates[0]
        result = env.step(xi)
        print("✓ Experiment step successful")
        
        # Test surrogate model
        surrogate = MPNNSurrogate(mocu_scale=1.0)
        mocu_pred = surrogate.forward_mocu(belief_graph)
        print("✓ Surrogate model forward pass successful")
        
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_data_generation():
    """Test data generation."""
    print("\nTesting data generation...")
    
    try:
        from data_generation.synthetic_data import SyntheticDataGenerator
        generator = SyntheticDataGenerator(N=3, K=2, n_samples=5)
        dataset = generator.generate_dataset(seed=42)
        print(f"✓ Generated {len(dataset)} samples")
        print(f"✓ Sample keys: {list(dataset[0].keys())}")
        return True
    except Exception as e:
        print(f"✗ Data generation test failed: {e}")
        return False

def test_evaluation():
    """Test evaluation pipeline."""
    print("\nTesting evaluation pipeline...")
    
    try:
        from eval.run_eval import run_episode
        from design.greedy_erm import choose_next_pair_greedy
        from core.kuramoto_env import PairTestEnv
        from surrogate.mpnn_surrogate import MPNNSurrogate
        
        # Create test environment with surrogate
        N, K = 3, 2
        omega = np.random.uniform(-1.0, 1.0, size=N)
        prior_bounds = (0.05, 0.50)
        surrogate = MPNNSurrogate(mocu_scale=1.0)
        env = PairTestEnv(N=N, omega=omega, prior_bounds=prior_bounds, K=K, surrogate=surrogate)
        
        # Test episode
        sim_opts = {'dt': 0.01, 'T': 5.0, 'burn_in': 2.0, 'R_target': 0.95}
        result = run_episode(env, choose_next_pair_greedy, sim_opts)
        
        print(f"✓ Episode completed: a_ctrl* = {result['a_ctrl_star']:.4f}")
        return True
    except Exception as e:
        print(f"✗ Evaluation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("TESTING DAD MOCU KURAMOTO DEMO")
    print("="*60)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_data_generation,
        test_evaluation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("="*60)
    print(f"TESTS COMPLETED: {passed}/{total} passed")
    print("="*60)
    
    if passed == total:
        print("🎉 All tests passed! The demo should work correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
