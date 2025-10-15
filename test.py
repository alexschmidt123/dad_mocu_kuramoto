#!/usr/bin/env python3
"""
Evaluation script with progress bars for Kuramoto experiment design methods.
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
from design.train_rl import make_hist_tokens
from eval.run_eval import run_episode
from eval.metrics import (print_comparison_results, plot_mocu_curves, 
                         plot_a_ctrl_distribution, save_results, 
                         compute_improvement_metrics)

# Try to import tqdm
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def run_multiple_episodes_with_progress(env_factory, chooser_fn, sim_opts, n_episodes=10, 
                                       strategy_name="Strategy", verbose=False, seed=None):
    """Run multiple episodes with progress bar."""
    if seed is not None:
        np.random.seed(seed)
    
    results = []
    
    if TQDM_AVAILABLE:
        pbar = tqdm(range(n_episodes), desc=f"Evaluating {strategy_name}", 
                   unit="episode", ncols=100,
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for episode in pbar:
            env = env_factory()
            result = run_episode(env, chooser_fn, sim_opts, verbose=verbose)
            results.append(result)
        
        pbar.close()
    else:
        print(f"\nEvaluating {strategy_name}: {n_episodes} episodes")
        for episode in range(n_episodes):
            if (episode + 1) % 10 == 0 or (episode + 1) == n_episodes:
                progress = (episode + 1) / n_episodes * 100
                print(f"  [{progress:5.1f}%] {episode+1}/{n_episodes} episodes")
            
            env = env_factory()
            result = run_episode(env, chooser_fn, sim_opts, verbose=verbose)
            results.append(result)
    
    return results


def compute_statistics(results: List[Dict]) -> Dict[str, Any]:
    """Compute statistics from multiple episode results."""
    a_ctrl_stars = [r['a_ctrl_star'] for r in results]
    terminal_mocus = [r['terminal_mocu_hat'] for r in results if r['terminal_mocu_hat'] is not None]
    total_times = [r['total_time'] for r in results]
    
    stats = {
        'n_episodes': len(results),
        'a_ctrl_star': {
            'mean': np.mean(a_ctrl_stars),
            'std': np.std(a_ctrl_stars),
            'min': np.min(a_ctrl_stars),
            'max': np.max(a_ctrl_stars),
            'median': np.median(a_ctrl_stars)
        },
        'terminal_mocu': {
            'mean': np.mean(terminal_mocus) if terminal_mocus else None,
            'std': np.std(terminal_mocus) if terminal_mocus else None,
            'min': np.min(terminal_mocus) if terminal_mocus else None,
            'max': np.max(terminal_mocus) if terminal_mocus else None,
            'median': np.median(terminal_mocus) if terminal_mocus else None
        },
        'total_time': {
            'mean': np.mean(total_times),
            'std': np.std(total_times),
            'min': np.min(total_times),
            'max': np.max(total_times)
        }
    }
    
    return stats


def compare_strategies_with_progress(env_factory, strategies: Dict, sim_opts, 
                                     n_episodes=10, verbose=False, seed=None):
    """Compare multiple strategies with progress bars."""
    if seed is not None:
        np.random.seed(seed)
    
    comparison_results = {}
    
    print("\n" + "="*80)
    print(f"RUNNING EVALUATION ({n_episodes} episodes per strategy)")
    print("="*80)
    
    for strategy_name, chooser_fn in strategies.items():
        results = run_multiple_episodes_with_progress(
            env_factory, chooser_fn, sim_opts, 
            n_episodes, strategy_name=strategy_name, 
            verbose=False, seed=None
        )
        stats = compute_statistics(results)
        comparison_results[strategy_name] = {
            'results': results,
            'statistics': stats
        }
    
    return comparison_results


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_env_factory(cfg: Dict, surrogate, seed=0):
    """Create environment factory."""
    episode_counter = [0]  # Use list to make it mutable
    
    def env_factory():
        N, K = cfg["N"], cfg["K"]
        # Use different seed for each episode
        episode_seed = cfg["seed"] + seed + episode_counter[0]
        episode_counter[0] += 1
        rng = np.random.default_rng(episode_seed)
        
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
        print(f"WARNING: Models directory not found: {models_dir}")
        print("Run train.py first to generate models")
        return {}
    
    available_models = {}
    
    # Check for MPNN Surrogate
    surrogate_path = os.path.join(models_dir, "mpnn_surrogate.pth")
    if os.path.exists(surrogate_path):
        print(f"Found: mpnn_surrogate.pth")
        
        # Try to load with config parameters first
        try:
            surrogate = MPNNSurrogate(
                mocu_scale=cfg["surrogate"].get("mocu_scale", 1.0),
                hidden=cfg["surrogate"]["hidden"],
                dropout=cfg["surrogate"]["dropout"]
            )
            surrogate.load_state_dict(torch.load(surrogate_path, map_location=device, weights_only=True))
            print(f"✓ Loaded with config parameters (hidden={cfg['surrogate']['hidden']})")
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"⚠ Architecture mismatch detected. Trying to detect saved model architecture...")
                
                # Load the state dict to inspect the architecture
                state_dict = torch.load(surrogate_path, map_location=device, weights_only=True)
                
                # Detect hidden size from the first layer
                first_layer_weight = state_dict['node_encoder.net.0.weight']
                detected_hidden = first_layer_weight.shape[0]
                
                print(f"Detected saved model architecture: hidden={detected_hidden}")
                print(f"Config architecture: hidden={cfg['surrogate']['hidden']}")
                print(f"Creating model with detected architecture...")
                
                # Create model with detected architecture
                surrogate = MPNNSurrogate(
                    mocu_scale=cfg["surrogate"].get("mocu_scale", 1.0),
                    hidden=detected_hidden,
                    dropout=cfg["surrogate"]["dropout"]
                )
                surrogate.load_state_dict(state_dict)
                print(f"✓ Loaded with detected parameters (hidden={detected_hidden})")
            else:
                raise e
        
        surrogate.to(device)
        surrogate.eval()
        available_models["surrogate"] = surrogate
    else:
        print(f"Not found: mpnn_surrogate.pth")
    
    # Check for Fixed Design
    fixed_path = os.path.join(models_dir, "fixed_design.pkl")
    if os.path.exists(fixed_path):
        print(f"Found: fixed_design.pkl")
        with open(fixed_path, 'rb') as f:
            fixed_sequence = pickle.load(f)
        available_models["fixed_design"] = fixed_sequence
    else:
        print(f"Not found: fixed_design.pkl")
    
    # Check for DAD Policy
    dad_path = os.path.join(models_dir, "dad_policy.pth")
    if os.path.exists(dad_path):
        print(f"Found: dad_policy.pth")
        policy = DADPolicy(hidden=cfg["dad_bc"]["hidden"])
        policy.load_state_dict(torch.load(dad_path, map_location='cpu'))
        policy.eval()
        available_models["dad_policy"] = policy
    else:
        print(f"Not found: dad_policy.pth")
    
    print("="*80)
    return available_models


def build_strategies(available_models: Dict) -> Dict:
    """Build strategy dictionary from available models."""
    strategies = {}
    
    strategies["Random"] = random_chooser
    
    if "fixed_design" in available_models:
        strategies["Fixed Design"] = fixed_design_chooser_factory(
            available_models["fixed_design"]
        )
    
    if "surrogate" in available_models:
        strategies["Greedy MPNN"] = greedy_chooser
    
    if "surrogate" in available_models and "dad_policy" in available_models:
        strategies["DAD"] = dad_chooser_factory(available_models["dad_policy"])
    
    return strategies


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained models for Kuramoto experiment design"
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Config file")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--models-dir", default="models", help="Models directory")
    parser.add_argument("--save-results", help="Save results to JSON file")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    
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
        print("\nERROR: No models found!")
        print("Run train.py first:")
        print("  python train.py --methods all")
        return 1
    
    # Build strategies
    strategies = build_strategies(available_models)
    
    print("\n" + "="*80)
    print("EVALUATION STRATEGIES")
    print("="*80)
    print(f"Available strategies: {list(strategies.keys())}")
    print("="*80)
    
    # Create environment factory
    surrogate = available_models.get("surrogate", None)
    env_factory = make_env_factory(cfg, surrogate=surrogate)
    
    # Run evaluation with progress bars
    try:
        comparison_results = compare_strategies_with_progress(
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
            print("Saved: mocu_curves.png")
            
            plot_a_ctrl_distribution(comparison_results, 
                                    save_path="a_ctrl_distribution.png")
            print("Saved: a_ctrl_distribution.png")
        except Exception as e:
            print(f"WARNING: Could not generate plots: {e}")
        
        # Save results
        if args.save_results:
            save_results(comparison_results, args.save_results)
            print(f"Saved: {args.save_results}")
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n\nERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        exit(1)