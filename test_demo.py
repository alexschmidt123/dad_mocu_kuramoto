#!/usr/bin/env python3
"""
Comprehensive test suite for the fixed Kuramoto experiment design project.
Tests all major components and their integration.
"""

import sys
import os
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all modules can be imported."""
    print("="*80)
    print("TEST 1: Module Imports")
    print("="*80)
    
    try:
        from core.kuramoto_env import PairTestEnv
        from core.belief import init_history, update_intervals, pair_threshold, build_belief_graph
        from core.pacemaker_control import simulate_with_pacemaker, sync_check
        from core.bisection import find_min_a_ctrl
        from surrogate.mpnn_surrogate import MPNNSurrogate
        from design.greedy_erm import choose_next_pair_greedy
        from design.dad_policy import DADPolicy
        from eval.run_eval import run_episode
        print("✓ All imports successful\n")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_belief_system():
    """Test belief representation and updates."""
    print("="*80)
    print("TEST 2: Belief System")
    print("="*80)
    
    try:
        from core.belief import init_history, update_intervals, pair_threshold, build_belief_graph
        
        N = 5
        omega = np.array([0.5, -0.3, 0.8, -0.1, 0.2])
        prior_bounds = (0.05, 0.50)
        
        # Initialize history
        h = init_history(N, prior_bounds)
        print(f"✓ Initialized history for N={N}")
        print(f"  Initial intervals: [{h.lower[0,1]:.3f}, {h.upper[0,1]:.3f}]")
        
        # Test pair threshold
        lam = pair_threshold(omega, 0, 1)
        expected_lam = 0.5 * abs(omega[0] - omega[1])
        assert abs(lam - expected_lam) < 1e-6, "Pair threshold calculation error"
        print(f"✓ Pair threshold λ_{{0,1}} = {lam:.4f}")
        
        # Test belief update
        xi = (0, 1)
        y_sync = True
        update_intervals(h, xi, y_sync, omega)
        print(f"✓ Updated belief after observing sync")
        print(f"  New interval: [{h.lower[0,1]:.3f}, {h.upper[0,1]:.3f}]")
        assert h.lower[0,1] >= lam or abs(h.lower[0,1] - lam) < 1e-6, "Lower bound should increase"
        
        # Test belief graph construction
        belief_graph = build_belief_graph(h, omega)
        assert 'node_feats' in belief_graph
        assert 'edge_feats' in belief_graph
        assert 'edge_index' in belief_graph
        print(f"✓ Built belief graph")
        print(f"  Nodes: {belief_graph['node_feats'].shape}")
        print(f"  Edges: {belief_graph['edge_feats'].shape}")
        
        print()
        return True
    except Exception as e:
        print(f"✗ Belief system test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_pacemaker_control():
    """Test pacemaker control simulation."""
    print("="*80)
    print("TEST 3: Pacemaker Control")
    print("="*80)
    
    try:
        from core.pacemaker_control import simulate_with_pacemaker, sync_check
        
        N = 5
        omega = np.random.uniform(-1.0, 1.0, N)
        A = np.random.uniform(0.1, 0.3, (N, N))
        A = 0.5 * (A + A.T)
        np.fill_diagonal(A, 0.0)
        
        # Test simulation
        print("Testing simulation...")
        result = simulate_with_pacemaker(A, omega, a_ctrl=0.2, T=3.0, method='Euler', n_trajectories=2)
        assert 't' in result
        assert 'R' in result
        assert 'R_mean' in result
        print(f"✓ Simulation successful")
        print(f"  Mean R = {result['R_mean']:.4f}")
        print(f"  Final R = {result['R'][-1]:.4f}")
        
        # Test sync check
        print("Testing sync check...")
        a_ctrl_low = 0.05
        a_ctrl_high = 1.0
        sync_low = sync_check(A, omega, a_ctrl_low, T=3.0, method='Euler', n_trajectories=2)
        sync_high = sync_check(A, omega, a_ctrl_high, T=3.0, method='Euler', n_trajectories=2)
        print(f"✓ Sync check successful")
        print(f"  a_ctrl={a_ctrl_low:.2f}: {sync_low}")
        print(f"  a_ctrl={a_ctrl_high:.2f}: {sync_high}")
        
        print()
        return True
    except Exception as e:
        print(f"✗ Pacemaker control test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_bisection():
    """Test bisection search."""
    print("="*80)
    print("TEST 4: Bisection Search")
    print("="*80)
    
    try:
        from core.bisection import find_min_a_ctrl
        
        # Simple test: find sqrt(2)
        def check_fn(x):
            return x**2 >= 2.0
        
        result = find_min_a_ctrl(None, None, check_fn, lo=0.0, hi_init=1.0, tol=1e-4)
        expected = np.sqrt(2)
        error = abs(result - expected)
        
        print(f"✓ Bisection search successful")
        print(f"  Found: {result:.6f}")
        print(f"  Expected: {expected:.6f}")
        print(f"  Error: {error:.2e}")
        
        assert error < 1e-3, f"Bisection error too large: {error}"
        
        print()
        return True
    except Exception as e:
        print(f"✗ Bisection test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_surrogate_model():
    """Test surrogate model."""
    print("="*80)
    print("TEST 5: Surrogate Model")
    print("="*80)
    
    try:
        from surrogate.mpnn_surrogate import MPNNSurrogate
        from core.belief import init_history, build_belief_graph
        
        # Create test data
        N = 5
        omega = np.random.uniform(-1.0, 1.0, N)
        h = init_history(N, (0.05, 0.50))
        belief_graph = build_belief_graph(h, omega)
        
        # Create model
        model = MPNNSurrogate(mocu_scale=1.0, hidden=32)
        print(f"✓ Created model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test MOCU prediction
        mocu_pred = model.forward_mocu(belief_graph)
        assert mocu_pred.numel() == 1
        assert mocu_pred.item() >= 0, "MOCU should be non-negative"
        print(f"✓ MOCU prediction: {mocu_pred.item():.4f}")
        
        # Test ERM prediction
        xi = (0, 1)
        erm_pred = model.forward_erm(belief_graph, xi)
        assert erm_pred.numel() == 1
        assert erm_pred.item() >= 0, "ERM should be non-negative"
        print(f"✓ ERM prediction for pair {xi}: {erm_pred.item():.4f}")
        
        # Test Sync prediction
        A_min = h.lower.copy()
        a_ctrl = 0.2
        sync_pred = model.forward_sync(A_min, a_ctrl, belief_graph)
        assert sync_pred.numel() == 1
        assert 0 <= sync_pred.item() <= 1, "Sync prob should be in [0,1]"
        print(f"✓ Sync prediction (a_ctrl={a_ctrl}): {sync_pred.item():.4f}")
        
        # Test gradient flow
        loss = mocu_pred + erm_pred + sync_pred
        loss.backward()
        print(f"✓ Gradient flow successful")
        
        print()
        return True
    except Exception as e:
        print(f"✗ Surrogate model test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_environment():
    """Test experiment environment."""
    print("="*80)
    print("TEST 6: Experiment Environment")
    print("="*80)
    
    try:
        from core.kuramoto_env import PairTestEnv
        from surrogate.mpnn_surrogate import MPNNSurrogate
        
        N, K = 5, 3
        omega = np.random.uniform(-1.0, 1.0, N)
        prior_bounds = (0.05, 0.50)
        surrogate = MPNNSurrogate(mocu_scale=1.0)
        
        env = PairTestEnv(N=N, omega=omega, prior_bounds=prior_bounds, K=K, surrogate=surrogate)
        print(f"✓ Created environment (N={N}, K={K})")
        
        # Test candidate pairs
        candidates = env.candidate_pairs()
        expected_n_pairs = N * (N - 1) // 2
        assert len(candidates) == expected_n_pairs, f"Expected {expected_n_pairs} pairs, got {len(candidates)}"
        print(f"✓ Found {len(candidates)} candidate pairs")
        
        # Test experiment step
        xi = candidates[0]
        result = env.step(xi)
        assert 'y' in result
        assert 'h' in result
        print(f"✓ Executed experiment on pair {xi}")
        print(f"  Outcome: {'sync' if result['y'] else 'not sync'}")
        
        # Test belief graph
        belief_graph = env.features()
        assert belief_graph['node_feats'].shape[0] == N
        print(f"✓ Extracted belief graph features")
        
        print()
        return True
    except Exception as e:
        print(f"✗ Environment test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_strategies():
    """Test experiment selection strategies."""
    print("="*80)
    print("TEST 7: Experiment Selection Strategies")
    print("="*80)
    
    try:
        from core.kuramoto_env import PairTestEnv
        from surrogate.mpnn_surrogate import MPNNSurrogate
        from design.greedy_erm import choose_next_pair_greedy
        
        # Create environment
        N, K = 5, 3
        omega = np.random.uniform(-1.0, 1.0, N)
        surrogate = MPNNSurrogate(mocu_scale=1.0)
        env = PairTestEnv(N=N, omega=omega, prior_bounds=(0.05, 0.50), K=K, surrogate=surrogate)
        
        candidates = env.candidate_pairs()
        
        # Test random strategy
        random_pair = candidates[np.random.randint(len(candidates))]
        print(f"✓ Random strategy selected: {random_pair}")
        
        # Test greedy ERM strategy
        greedy_pair = choose_next_pair_greedy(env, candidates)
        assert greedy_pair in candidates
        print(f"✓ Greedy ERM strategy selected: {greedy_pair}")
        
        print()
        return True
    except Exception as e:
        print(f"✗ Strategies test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation():
    """Test evaluation pipeline."""
    print("="*80)
    print("TEST 8: Evaluation Pipeline")
    print("="*80)
    
    try:
        from eval.run_eval import run_episode
        from core.kuramoto_env import PairTestEnv
        from surrogate.mpnn_surrogate import MPNNSurrogate
        from design.greedy_erm import choose_next_pair_greedy
        
        # Create environment
        N, K = 4, 2
        omega = np.random.uniform(-1.0, 1.0, N)
        surrogate = MPNNSurrogate(mocu_scale=1.0)
        env = PairTestEnv(N=N, omega=omega, prior_bounds=(0.05, 0.50), K=K, surrogate=surrogate)
        
        # Run episode
        sim_opts = {'dt': 0.01, 'T': 3.0, 'burn_in': 1.5, 'R_target': 0.95, 'method': 'Euler', 'n_trajectories': 2}
        
        print("Running evaluation episode...")
        result = run_episode(env, choose_next_pair_greedy, sim_opts, verbose=False)
        
        assert 'a_ctrl_star' in result
        assert 'terminal_mocu_hat' in result
        assert 'intermediate_results' in result
        assert len(result['intermediate_results']) == K
        
        print(f"✓ Episode completed successfully")
        print(f"  Final a_ctrl*: {result['a_ctrl_star']:.4f}")
        if result['terminal_mocu_hat'] is not None:
            print(f"  Terminal MOCU: {result['terminal_mocu_hat']:.4f}")
        print(f"  Steps executed: {len(result['intermediate_results'])}")
        
        print()
        return True
    except Exception as e:
        print(f"✗ Evaluation test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_dad_policy():
    """Test DAD policy."""
    print("="*80)
    print("TEST 9: DAD Policy")
    print("="*80)
    
    try:
        from design.dad_policy import DADPolicy
        from core.kuramoto_env import PairTestEnv
        from surrogate.mpnn_surrogate import MPNNSurrogate
        
        # Create policy
        policy = DADPolicy(hidden=32)
        print(f"✓ Created DAD policy with {sum(p.numel() for p in policy.parameters())} parameters")
        
        # Create environment
        N, K = 5, 3
        omega = np.random.uniform(-1.0, 1.0, N)
        surrogate = MPNNSurrogate(mocu_scale=1.0)
        env = PairTestEnv(N=N, omega=omega, prior_bounds=(0.05, 0.50), K=K, surrogate=surrogate)
        
        # Test forward pass
        belief_graph = env.features()
        candidates = env.candidate_pairs()
        hist_tokens = []  # Empty history initially
        
        scores = policy.forward(belief_graph, candidates, hist_tokens, N)
        assert scores.shape[0] == len(candidates)
        print(f"✓ Forward pass successful")
        print(f"  Score shape: {scores.shape}")
        print(f"  Score range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
        
        # Test choose method
        chosen_pair = policy.choose(env, hist_tokens)
        assert chosen_pair in candidates
        print(f"✓ Policy chose pair: {chosen_pair}")
        
        print()
        return True
    except Exception as e:
        print(f"✗ DAD policy test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST SUITE FOR KURAMOTO EXPERIMENT DESIGN")
    print("="*80 + "\n")
    
    tests = [
        ("Module Imports", test_imports),
        ("Belief System", test_belief_system),
        ("Pacemaker Control", test_pacemaker_control),
        ("Bisection Search", test_bisection),
        ("Surrogate Model", test_surrogate_model),
        ("Experiment Environment", test_environment),
        ("Selection Strategies", test_strategies),
        ("Evaluation Pipeline", test_evaluation),
        ("DAD Policy", test_dad_policy),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"✗ Test '{name}' crashed: {e}\n")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status:12} {name}")
    
    print("="*80)
    print(f"RESULTS: {passed_count}/{total_count} tests passed")
    print("="*80 + "\n")
    
    if passed_count == total_count:
        print("🎉 All tests passed! The project is working correctly.")
        print("\nNext steps:")
        print("  1. Run quick demo: python main_demo.py --mode quick")
        print("  2. Run full evaluation: python main_demo.py --mode full --episodes 10")
        return 0
    else:
        print("❌ Some tests failed. Please review the errors above.")
        print("\nCommon issues:")
        print("  - Missing scipy: pip install scipy")
        print("  - Missing torch: pip install torch")
        print("  - Wrong working directory: cd to project root")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)