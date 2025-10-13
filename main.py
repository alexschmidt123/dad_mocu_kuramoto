#!/usr/bin/env python3
"""
Unified pipeline for Kuramoto experiment design with Deep Adaptive Design.
RTX 4090 optimized with CUDA, AMP, and parallel data generation.
"""

import yaml
import numpy as np
import torch
import argparse
import os
import warnings
from typing import Dict, Any

from core.kuramoto_env import PairTestEnv
from surrogate.mpnn_surrogate import MPNNSurrogate
from surrogate.train_surrogate import train_surrogate_model
from design.greedy_erm import choose_next_pair_greedy
from design.dad_policy import DADPolicy
from design.train_bc import train_behavior_cloning, make_hist_tokens
from eval.run_eval import compare_strategies
from eval.metrics import (print_comparison_results, plot_mocu_curves, 
                         plot_a_ctrl_distribution, save_results, 
                         compute_improvement_metrics)

warnings.filterwarnings('ignore', category=UserWarning)


def setup_gpu():
    """Configure GPU for optimal performance."""
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Install with:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return False
    
    # Enable cuDNN benchmarking
    torch.backends.cudnn.benchmark = True
    
    # Enable TF32 on Ampere GPUs for faster matmul
    if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True
    
    print("="*80)
    print("GPU Configuration")
    print("="*80)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    print(f"Mixed Precision: Enabled")
    print(f"cuDNN Benchmark: Enabled")
    print("="*80)
    return True


def load_config(path: str = "configs/config.yaml") -> Dict[str, Any]:
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
    """Random baseline."""
    return cands[np.random.randint(len(cands))]


def fixed_design_chooser_factory(fixed_sequence):
    """Fixed (static) design: pre-determined experiment sequence."""
    step_counter = [0]  # Use list to maintain state across calls
    
    def fixed_chooser(env, cands):
        if step_counter[0] < len(fixed_sequence):
            chosen = fixed_sequence[step_counter[0]]
            step_counter[0] += 1
            # Ensure the pair is in candidates
            if chosen in cands:
                return chosen
            else:
                # Fallback to first candidate if pre-determined pair not available
                return cands[0]
        else:
            # Fallback to random if we run out of fixed designs
            return cands[np.random.randint(len(cands))]
    
    return fixed_chooser


def greedy_chooser(env, cands):
    """Greedy ERM baseline."""
    return choose_next_pair_greedy(env, cands)


def dad_chooser_factory(policy: DADPolicy):
    """Create DAD chooser."""
    def dad_chooser(env, cands):
        hist_tokens = make_hist_tokens(env.h, env.N)
        return policy.choose(env, hist_tokens)
    return dad_chooser


def main():
    parser = argparse.ArgumentParser(
        description="Deep Adaptive Design for Kuramoto Networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (10 episodes, ~2 minutes)
  python main.py --mode quick
  
  # Full evaluation (50 episodes, ~8 minutes)
  python main.py --mode full --episodes 50
  
  # Custom config
  python main.py --config my_config.yaml --episodes 100
  
  # Save results
  python main.py --mode full --save-results results.json
        """
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Config file")
    parser.add_argument("--mode", choices=["quick", "full"], default="full",
                       help="Evaluation mode")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--save-results", help="Save results to JSON")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    
    args = parser.parse_args()
    
    # Setup
    if not args.cpu:
        gpu_ok = setup_gpu()
    else:
        gpu_ok = False
        print("Running in CPU mode")
    
    device = 'cuda' if gpu_ok else 'cpu'
    
    # Load config
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found!")
        return 1
    
    cfg = load_config(args.config)
    
    # Adjust for quick mode
    if args.mode == "quick":
        cfg["surrogate"]["n_train"] = 500
        cfg["surrogate"]["n_val"] = 100
        cfg["surrogate"]["n_theta_samples"] = 10
        cfg["surrogate"]["epochs"] = 30
        cfg["dad_bc"]["epochs"] = 3
        args.episodes = 10
        print("\nQuick mode: reduced training for fast testing\n")
    
    print("\n" + "="*80)
    print("UNIFIED PIPELINE: Data Generation → Training → Evaluation")
    print("="*80)
    print(f"Configuration: N={cfg['N']}, K={cfg['K']}, Episodes={args.episodes}")
    print(f"Device: {device.upper()}")
    print("="*80 + "\n")
    
    # Step 1: Train Surrogate Model
    print("STEP 1/3: Training MPNN Surrogate Model")
    print("="*80)
    surrogate, _ = train_surrogate_model(
        N=cfg["N"],
        K=cfg["K"],
        n_train=cfg["surrogate"]["n_train"],
        n_val=cfg["surrogate"]["n_val"],
        n_theta_samples=cfg["surrogate"]["n_theta_samples"],
        epochs=cfg["surrogate"]["epochs"],
        lr=cfg["surrogate"]["lr"],
        batch_size=cfg["surrogate"]["batch_size"],
        device=device,
        save_path="trained_surrogate.pth"
    )
    print("\n✓ Surrogate training complete\n")
    
    # Step 2: Train DAD Policy & Generate Fixed Design
    if cfg.get("enable_dad", True):
        print("STEP 2/3: Training DAD Policy via Behavior Cloning")
        print("="*80)
        
        env_factory = make_env_factory(cfg, surrogate=surrogate)
        policy = DADPolicy(hidden=cfg["dad_bc"]["hidden"])
        
        policy = train_behavior_cloning(
            env_factory=env_factory,
            policy=policy,
            epochs=cfg["dad_bc"]["epochs"],
            episodes_per_epoch=cfg["dad_bc"]["episodes_per_epoch"],
            lr=cfg["dad_bc"]["lr"]
        )
        
        torch.save(policy.state_dict(), "trained_dad_policy.pth")
        print("\n✓ DAD policy training complete\n")
        
        # Generate fixed design sequence (optimal for average case)
        print("Generating fixed design sequence...")
        representative_env = env_factory()
        fixed_sequence = []
        for _ in range(cfg["K"]):
            cands = representative_env.candidate_pairs()
            xi = choose_next_pair_greedy(representative_env, cands)
            fixed_sequence.append(xi)
            representative_env.step(xi)
        print(f"✓ Fixed sequence: {fixed_sequence}\n")
        
        strategies = {
            "Random": random_chooser,
            "Fixed Design": fixed_design_chooser_factory(fixed_sequence),
            "Greedy MPNN": greedy_chooser,
            "DAD": dad_chooser_factory(policy),
        }
    else:
        # Generate fixed design even without DAD
        print("Generating fixed design sequence...")
        representative_env = make_env_factory(cfg, surrogate=surrogate)()
        fixed_sequence = []
        for _ in range(cfg["K"]):
            cands = representative_env.candidate_pairs()
            xi = choose_next_pair_greedy(representative_env, cands)
            fixed_sequence.append(xi)
            representative_env.step(xi)
        print(f"✓ Fixed sequence: {fixed_sequence}\n")
        
        strategies = {
            "Random": random_chooser,
            "Fixed Design": fixed_design_chooser_factory(fixed_sequence),
            "Greedy MPNN": greedy_chooser,
        }
    
    # Step 3: Comprehensive Evaluation
    print("STEP 3/3: Comprehensive Evaluation")
    print("="*80)
    print(f"Running {args.episodes} episodes per strategy...\n")
    
    env_factory = make_env_factory(cfg, surrogate=surrogate)
    
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
    if "Random" in comparison_results and "Greedy MPNN" in comparison_results:
        print("\n" + "="*80)
        print("IMPROVEMENT vs Random Baseline")
        print("="*80)
        improvements = compute_improvement_metrics(comparison_results, baseline_strategy="Random")
        for strategy, metrics in improvements.items():
            print(f"\n{strategy}:")
            if 'mocu_improvement_pct' in metrics:
                print(f"  MOCU: {metrics['mocu_improvement_pct']:+.1f}%")
            print(f"  a_ctrl: {metrics['a_ctrl_improvement_pct']:+.1f}%")
    
    # Generate plots
    print("\n" + "="*80)
    print("Generating Plots")
    print("="*80)
    try:
        plot_mocu_curves(comparison_results, save_path="mocu_curves.png")
        print("✓ Saved MOCU curves to mocu_curves.png")
        plot_a_ctrl_distribution(comparison_results, save_path="a_ctrl_distribution.png")
        print("✓ Saved control parameter distribution to a_ctrl_distribution.png")
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
    
    # Save results
    if args.save_results:
        save_results(comparison_results, args.save_results)
        print(f"✓ Saved results to {args.save_results}")
    
    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETE")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)