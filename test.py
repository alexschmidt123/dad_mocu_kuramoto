#!/usr/bin/env python3
"""
Evaluation script for Kuramoto experiment design methods.

Automatically scans models/ directory and evaluates all available methods:
- Random (no model needed)
- Fixed Design (if fixed_design.pkl exists)
- Greedy MPNN (if mpnn_surrogate.pth exists)
- DAD (if dad_policy.pth exists)

Usage:
  # Evaluate all models in models/ directory
  python test.py --config configs/config.yaml --episodes 50
  
  # Quick test
  python test.py --config configs/config_fast.yaml --episodes 10
  
  # Save results
  python test.py --episodes 50 --save-results results.json
  
  # Custom models directory
  python test.py --models-dir my_models/ --episodes 50
"""

import yaml
import numpy as np
import torch
import argparse
import os
import pickle
from typing import Dict, Any, List

from core.kuramoto_env import PairTestEnv
from surrogate.mpnn_surrogate import MPNNSurrogate
from design.greedy_erm import choose_next_pair_greedy
from design.dad_policy import DADPolicy
from design.train_bc import make_hist_tokens
from eval.run_eval import compare_strategies
from eval.metrics import (print_comparison_results, plot_mocu_curves, 
                         plot_a_ctrl_distribution, save_results, 
                         compute_improvement_metrics)


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_env_factory(cfg: Dict, surrogate, seed=0):
    """Create environment factory."""
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


def random_chooser(env, cands):
    """Random baseline (no model needed)."""
    return cands[np.random.randint(len(cands))]


def fixed_design_chooser_factory(fixed_sequence):
    """Fixed design chooser."""
    step_counter = [0]
    
    def fixed_chooser(env, cands):
        if step_counter[0] < len(fixed_sequence):
            chosen = fixed_sequence[step_counter[0]]
            step_counter[0] += 1
            if chosen in cands:
                return chosen
            else:
                return cands[0]
        else:
            return cands[np.random.randint(len(cands))]
    
    return fixed_chooser


def greedy_chooser(env, cands):
    """Greedy MPNN chooser."""
    return choose_next_pair_greedy(env, cands)


def dad_chooser_factory(policy: DADPolicy):
    """DAD policy chooser."""
    def dad_chooser(env, cands):
        hist_tokens = make_hist_tokens(env.h, env.N)
        return policy.choose(env, hist_tokens)
    return dad_chooser


def scan_models(models_dir: str, cfg: Dict, device: str) -> Dict:
    """Scan models directory and load available models."""
    print("\n" + "="*80)
    print("SCANNING MODELS")
    print("="*80)
    print(f"Models directory: {models_dir}")
    
    if not os.path.exists(models_dir):
        print(f"⚠ Models directory not found: {models_dir}")
        print("Run train.py first to generate models")
        return {}
    
    available_models = {}
    
    # Check for MPNN Surrogate
    surrogate_path = os.path.join(models_dir, "mpnn_surrogate.pth")
    if os.path.exists(surrogate_path):
        print(f"✓ Found: mpnn_surrogate.pth")
        surrogate = MPNNSurrogate(
            mocu_scale=cfg["surrogate"].get("mocu_scale", 1.0),
            hidden=cfg["surrogate"]["hidden"],
            dropout=cfg["surrogate"]["dropout"]
        )
        surrogate.load_state_dict(torch.load(surrogate_path, map_location=device))
        surrogate.to(device)
        surrogate.eval()
        available_models["surrogate"] = surrogate
    else:
        print(f"✗ Not found: mpnn_surrogate.pth")
        print("  Greedy MPNN and DAD cannot be evaluated without surrogate")
    
    # Check for Fixed Design
    fixed_path = os.path.join(models_dir, "fixed_design.pkl")
    if os.path.exists(fixed_path):
        print(f"✓ Found: fixed_design.pkl")
        with open(fixed_path, 'rb') as f:
            fixed_sequence = pickle.load(f)
        available_models["fixed_design"] = fixed_sequence
        print(f"  Sequence: {fixed_sequence}")
    else:
        print(f"✗ Not found: fixed_design.pkl")
    
    # Check for DAD Policy
    dad_path = os.path.join(models_dir, "dad_policy.pth")
    if os.path.exists(dad_path):
        print(f"✓ Found: dad_policy.pth")
        policy = DADPolicy(hidden=cfg["dad_bc"]["hidden"])
        policy.load_state_dict(torch.load(dad_path, map_location='cpu'))
        policy.eval()
        available_models["dad_policy"] = policy
    else:
        print(f"✗ Not found: dad_policy.pth")
    
    print("="*80)
    return available_models


def build_strategies(available_models: Dict) -> Dict:
    """Build strategy dictionary from available models."""
    strategies = {}
    
    # Random is always available
    strategies["Random"] = random_chooser
    
    # Fixed Design (if available)
    if "fixed_design" in available_models:
        strategies["Fixed Design"] = fixed_design_chooser_factory(
            available_models["fixed_design"]
        )
    
    # Greedy MPNN (needs surrogate)
    if "surrogate" in available_models:
        strategies["Greedy MPNN"] = greedy_chooser
    
    # DAD (needs surrogate and policy)
    if "surrogate" in available_models and "dad_policy" in available_models:
        strategies["DAD"] = dad_chooser_factory(available_models["dad_policy"])
    
    return strategies


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained models for Kuramoto experiment design",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all models (50 episodes)
  python test.py --episodes 50
  
  # Quick test (10 episodes)
  python test.py --episodes 10
  
  # Custom config and save results
  python test.py --config configs/config_fast.yaml --episodes 50 --save-results results.json
  
  # Use different models directory
  python test.py --models-dir trained_models/ --episodes 50
        """
    )
    parser.add_argument("--config", default="configs/config.yaml",
                       help="Config file")
    parser.add_argument("--episodes", type=int, default=50,
                       help="Number of evaluation episodes")
    parser.add_argument("--models-dir", default="models",
                       help="Directory containing trained models")
    parser.add_argument("--save-results", 
                       help="Save results to JSON file")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU mode")
    
    args = parser.parse_args()
    
    # Setup device
    if not args.cpu and torch.cuda.is_available():
        device = 'cuda'
        print("Using GPU")
    else:
        device = 'cpu'
        print("Using CPU")
    
    # Load config
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found!")
        return 1
    
    cfg = load_config(args.config)
    
    print("\n" + "="*80)
    print("EVALUATION PIPELINE")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Episodes: {args.episodes}")
    print(f"Models dir: {args.models_dir}")
    print(f"Device: {device.upper()}")
    print("="*80)
    
    # Scan and load models
    available_models = scan_models(args.models_dir, cfg, device)
    
    if not available_models:
        print("\n✗ No models found!")
        print("Run train.py first to generate models:")
        print("  python train.py --methods all")
        return 1
    
    # Build strategies
    strategies = build_strategies(available_models)
    
    print("\n" + "="*80)
    print("EVALUATION STRATEGIES")
    print("="*80)
    print(f"Available strategies: {list(strategies.keys())}")
    print("="*80)
    
    if len(strategies) == 1:
        print("\n⚠ Only Random baseline available.")
        print("Train more models with: python train.py --methods all")
        print("Continuing with Random only...\n")
    
    # Create environment factory
    surrogate = available_models.get("surrogate", None)
    env_factory = make_env_factory(cfg, surrogate=surrogate)
    
    # Run evaluation
    print("\n" + "="*80)
    print(f"RUNNING EVALUATION ({args.episodes} episodes per strategy)")
    print("="*80)
    
    try:
        comparison_results = compare_strategies(
            env_factory=env_factory,
            strategies=strategies,
            sim_opts=cfg["sim"],
            n_episodes=args.episodes,
            verbose=False,
            seed=42
        )
        
        # Print results
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print_comparison_results(comparison_results)
        
        # Compute improvements
        if "Random" in comparison_results and len(comparison_results) > 1:
            print("\n" + "="*80)
            print("IMPROVEMENT vs Random Baseline")
            print("="*80)
            improvements = compute_improvement_metrics(
                comparison_results, baseline_strategy="Random"
            )
            for strategy, metrics in improvements.items():
                print(f"\n{strategy}:")
                if 'mocu_improvement_pct' in metrics:
                    print(f"  MOCU: {metrics['mocu_improvement_pct']:+.1f}%")
                print(f"  a_ctrl: {metrics['a_ctrl_improvement_pct']:+.1f}%")
        
        # Generate plots
        print("\n" + "="*80)
        print("GENERATING PLOTS")
        print("="*80)
        try:
            plot_mocu_curves(comparison_results, save_path="mocu_curves.png")
            print("✓ Saved: mocu_curves.png")
            
            plot_a_ctrl_distribution(comparison_results, 
                                    save_path="a_ctrl_distribution.png")
            print("✓ Saved: a_ctrl_distribution.png")
        except Exception as e:
            print(f"⚠ Could not generate plots: {e}")
        
        # Save results
        if args.save_results:
            save_results(comparison_results, args.save_results)
            print(f"✓ Saved: {args.save_results}")
        
        print("\n" + "="*80)
        print("✓ EVALUATION COMPLETE")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        exit(1)