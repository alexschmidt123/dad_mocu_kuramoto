#!/usr/bin/env python3
"""
Fixed main demo for Kuramoto experiment design.
Uses corrected data generation and training procedures.
"""

import yaml
import numpy as np
import torch
import os
import argparse
from typing import Dict, Callable, Any
import warnings

from core.kuramoto_env import PairTestEnv
from surrogate.mpnn_surrogate import MPNNSurrogate
from surrogate.train_surrogate import train_surrogate_model
from design.greedy_erm import choose_next_pair_greedy
from design.dad_policy import DADPolicy
from design.train_bc import train_behavior_cloning
from eval.run_eval import run_episode, run_multiple_episodes, compare_strategies
from eval.metrics import (print_comparison_results, plot_mocu_curves, 
                         plot_a_ctrl_distribution, save_results, 
                         compute_improvement_metrics)
from data_generation.synthetic_data import SyntheticDataGenerator

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


def load_cfg(path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_env_factory(cfg: Dict[str, Any], surrogate=None, seed=0):
    """Create environment factory function."""
    def env_factory():
        N, K = cfg["N"], cfg["K"]
        rng = np.random.default_rng(cfg["seed"] + seed)
        
        if cfg["omega"]["kind"] == "uniform":
            omega = rng.uniform(cfg["omega"]["low"], cfg["omega"]["high"], size=N)
        else:
            omega = rng.normal(0.0, 1.0, size=N)
        
        prior = (cfg["prior_lower"], cfg["prior_upper"])
        return PairTestEnv(N=N, omega=omega, prior_bounds=prior, K=K, 
                          surrogate=surrogate, rng=rng)
    return env_factory


def greedy_chooser(env, cands):
    """Greedy ERM chooser."""
    return choose_next_pair_greedy(env, cands)


def random_chooser(env, cands):
    """Random chooser."""
    return cands[np.random.randint(len(cands))]


def dad_chooser_factory(policy: DADPolicy):
    """Create DAD chooser from trained policy."""
    def dad_chooser(env, cands):
        from design.train_bc import make_hist_tokens
        hist_tokens = make_hist_tokens(env.h, env.N)
        return policy.choose(env, hist_tokens)
    return dad_chooser


def train_surrogate_if_needed(cfg: Dict[str, Any], force_retrain: bool = False) -> MPNNSurrogate:
    """Train surrogate model if needed."""
    model_path = "trained_surrogate.pth"
    
    if os.path.exists(model_path) and not force_retrain:
        print(f"Loading pre-trained surrogate from {model_path}")
        try:
            model = MPNNSurrogate(mocu_scale=cfg["surrogate"].get("mocu_scale", 1.0))
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            print("✓ Model loaded successfully\n")
            return model
        except Exception as e:
            print(f"Warning: Could not load model ({e}). Training new model...\n")
    
    print("Training surrogate model (this may take a while)...")
    print("Using fixed data generation with proper MOCU/ERM computation\n")
    
    surrogate_cfg = cfg.get("surrogate", {})
    model, _ = train_surrogate_model(
        N=cfg["N"],
        K=cfg["K"],
        n_train=surrogate_cfg.get("n_train", 200),  # Reduced for demo
        n_val=surrogate_cfg.get("n_val", 50),
        n_theta_samples=surrogate_cfg.get("n_theta_samples", 10),
        epochs=surrogate_cfg.get("epochs", 30),
        lr=surrogate_cfg.get("lr", 1e-3),
        batch_size=surrogate_cfg.get("batch_size", 32),
        device='cpu',
        save_path=model_path
    )
    return model


def train_dad_policy(cfg: Dict[str, Any], surrogate: MPNNSurrogate) -> DADPolicy:
    """Train DAD policy using behavior cloning."""
    print("="*80)
    print("Training DAD policy via behavior cloning...")
    print("="*80)
    
    # Create environment factory for training
    env_factory = make_env_factory(cfg, surrogate=surrogate)
    
    # Initialize policy
    policy = DADPolicy(hidden=cfg.get("dad_bc", {}).get("hidden", 64))
    
    # Train using behavior cloning
    bc_config = cfg.get("dad_bc", {})
    trained_policy = train_behavior_cloning(
        env_factory=env_factory,
        policy=policy,
        epochs=bc_config.get("epochs", 3),
        episodes_per_epoch=bc_config.get("episodes_per_epoch", 20),
        lr=bc_config.get("lr", 1e-3)
    )
    
    print("✓ DAD policy training complete\n")
    return trained_policy


def run_comprehensive_evaluation(cfg: Dict[str, Any], n_episodes: int = 10, 
                                verbose: bool = False, save_results_path: str = None):
    """Run comprehensive evaluation comparing all strategies."""
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION")
    print("="*80)
    print(f"Configuration: N={cfg['N']}, K={cfg['K']}, Episodes={n_episodes}")
    print("="*80 + "\n")
    
    # Train surrogate model
    surrogate = train_surrogate_if_needed(cfg)
    
    # Create environment factory
    env_factory = make_env_factory(cfg, surrogate=surrogate)
    
    # Define strategies
    strategies = {
        "Random": random_chooser,
        "GreedyERM": greedy_chooser,
    }
    
    # Add DAD policy if enabled
    if cfg.get("enable_dad", True):
        try:
            dad_policy = train_dad_policy(cfg, surrogate)
            strategies["DAD"] = dad_chooser_factory(dad_policy)
        except Exception as e:
            print(f"Warning: Could not train DAD policy: {e}")
            print("Continuing with Random and GreedyERM only...\n")
    
    # Run comparison
    print("="*80)
    print(f"Running evaluation with {n_episodes} episodes per strategy...")
    print("="*80 + "\n")
    
    comparison_results = compare_strategies(
        env_factory=env_factory,
        strategies=strategies,
        sim_opts=cfg["sim"],
        n_episodes=n_episodes,
        verbose=verbose,
        seed=42
    )
    
    # Print results
    print_comparison_results(comparison_results)
    
    # Compute improvements
    if "Random" in comparison_results and "GreedyERM" in comparison_results:
        print("\n" + "="*80)
        print("IMPROVEMENT METRICS (vs Random Baseline)")
        print("="*80)
        improvements = compute_improvement_metrics(comparison_results, baseline_strategy="Random")
        for strategy, metrics in improvements.items():
            print(f"\n{strategy}:")
            if 'mocu_improvement_pct' in metrics:
                print(f"  MOCU improvement: {metrics['mocu_improvement_pct']:+.2f}%")
            print(f"  a_ctrl improvement: {metrics['a_ctrl_improvement_pct']:+.2f}%")
    
    # Generate plots
    try:
        print("\n" + "="*80)
        print("Generating plots...")
        print("="*80)
        plot_mocu_curves(comparison_results, save_path="mocu_curves.png")
        print("✓ Saved MOCU curves to mocu_curves.png")
        plot_a_ctrl_distribution(comparison_results, save_path="a_ctrl_distribution.png")
        print("✓ Saved control parameter distribution to a_ctrl_distribution.png")
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
    
    # Save results
    if save_results_path:
        save_results(comparison_results, save_results_path)
    
    return comparison_results


def run_quick_demo(cfg: Dict[str, Any]):
    """Run a quick demo with single episodes."""
    print("\n" + "="*80)
    print("QUICK DEMO")
    print("="*80)
    print(f"Configuration: N={cfg['N']}, K={cfg['K']}")
    print("="*80 + "\n")
    
    # Load or train surrogate
    surrogate = train_surrogate_if_needed(cfg)
    sim = cfg["sim"]
    
    # Test strategies
    strategies = {
        "Random": random_chooser,
        "GreedyERM": greedy_chooser,
    }
    
    results = {}
    for name, chooser in strategies.items():
        print(f"\n{'='*80}")
        print(f"Testing {name} Strategy")
        print(f"{'='*80}")
        
        env_factory = make_env_factory(cfg, surrogate=surrogate, 
                                       seed=0 if name=="GreedyERM" else 1)
        env = env_factory()
        out = run_episode(env, chooser, sim_opts=sim, verbose=True)
        
        results[name] = out
        print(f"\n{name} Results:")
        print(f"  Final a_ctrl*: {out['a_ctrl_star']:.4f}")
        print(f"  Terminal MOCU: {out['terminal_mocu_hat']:.4f}")
        print(f"  Total time: {out['total_time']:.2f}s")
    
    # Compare results
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    
    for metric in ['a_ctrl_star', 'terminal_mocu_hat']:
        print(f"\n{metric}:")
        for name in results:
            val = results[name].get(metric, 'N/A')
            if isinstance(val, float):
                print(f"  {name:12s}: {val:.4f}")
            else:
                print(f"  {name:12s}: {val}")


def main():
    parser = argparse.ArgumentParser(
        description="Fixed Kuramoto Experiment Design Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick demo with single episodes
  python main_demo.py --mode quick
  
  # Full evaluation with 20 episodes
  python main_demo.py --mode full --episodes 20
  
  # Force retrain surrogate model
  python main_demo.py --mode full --retrain
  
  # Save results to file
  python main_demo.py --mode full --episodes 10 --save-results results.json
        """
    )
    parser.add_argument("--config", default="configs/exp_fixedK.yaml", 
                       help="Config file path")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick",
                       help="Demo mode: quick (single episodes) or full (comprehensive)")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of episodes for full evaluation")
    parser.add_argument("--retrain", action="store_true",
                       help="Force retrain surrogate model")
    parser.add_argument("--save-results", help="Path to save results JSON")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output during evaluation")
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found!")
        return 1
    
    cfg = load_cfg(args.config)
    
    # Update config for retraining
    if args.retrain:
        print("Force retraining surrogate model...\n")
    
    try:
        if args.mode == "quick":
            run_quick_demo(cfg)
        else:
            run_comprehensive_evaluation(
                cfg, 
                n_episodes=args.episodes,
                verbose=args.verbose,
                save_results_path=args.save_results
            )
        
        print("\n" + "="*80)
        print("✓ Demo completed successfully!")
        print("="*80 + "\n")
        return 0
    
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())