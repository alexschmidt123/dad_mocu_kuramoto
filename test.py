#!/usr/bin/env python3
"""
Comprehensive evaluation of all design strategies with progress bars
Evaluates: Random, Fixed Design, Greedy MPNN, DAD Policy
"""

import yaml
import numpy as np
import torch
import argparse
import os
import pickle
import json
import time
from typing import Dict, Any, List

from core.kuramoto_env import PairTestEnv
from surrogate.mpnn_surrogate import MPNNSurrogate
from design.greedy_erm import choose_next_pair_greedy
from design.dad_policy import DADPolicy
from design.train_rl import make_hist_tokens
from core.bisection import find_min_a_ctrl
from core.pacemaker_control import sync_check

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration"""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_env_factory(cfg: Dict, surrogate, episode_counter=[0]):
    """Create environment factory with unique seeds per episode"""
    def env_factory():
        N, K = cfg["N"], cfg["K"]
        episode_seed = cfg["seed"] + episode_counter[0]
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
    """Random baseline"""
    return cands[np.random.randint(len(cands))]


def fixed_design_chooser_factory(fixed_sequence):
    """Fixed design chooser"""
    step_counter = [0]
    
    def fixed_chooser(env, cands):
        if step_counter[0] < len(fixed_sequence):
            chosen = fixed_sequence[step_counter[0]]
            step_counter[0] += 1
            if chosen in cands:
                return chosen
        return cands[0] if cands else (0, 1)
    
    return fixed_chooser

def make_env_factory(cfg: Dict, surrogate, episode_counter=[0]):
    """Create environment factory with unique seeds per episode"""
    def env_factory():
        N, K = cfg["N"], cfg["K"]
        episode_seed = cfg["seed"] + episode_counter[0]
        episode_counter[0] += 1
        rng = np.random.default_rng(episode_seed)
        
        # Match paper's omega range
        if cfg["omega"]["kind"] == "uniform":
            omega = rng.uniform(cfg["omega"]["low"], cfg["omega"]["high"], size=N)
        else:
            omega = rng.normal(0.0, 1.0, size=N)
        
        # CRITICAL: Use adaptive bounds
        aLower, aUpper = compute_adaptive_bounds_for_env(omega, N)
        prior = (aLower, aUpper)  # Now returns matrices
        
        return PairTestEnv(N=N, omega=omega, prior_bounds=prior, K=K, 
                          surrogate=surrogate, rng=rng)
    return env_factory

def compute_adaptive_bounds_for_env(omega, N):
    """Helper to compute adaptive bounds for env"""
    aUpper = np.zeros((N, N))
    aLower = np.zeros((N, N))
    
    for i in range(N):
        for j in range(i + 1, N):
            syncThreshold = 0.5 * np.abs(omega[i] - omega[j])
            aUpper[i, j] = syncThreshold * 1.15
            aLower[i, j] = syncThreshold * 0.85
            aUpper[j, i] = aUpper[i, j]
            aLower[j, i] = aLower[i, j]
    
    if N == 5:
        for i in [0]:
            for j in range(2, 5):
                aUpper[i, j] *= 0.3
                aLower[i, j] *= 0.3
                aUpper[j, i] = aUpper[i, j]
                aLower[j, i] = aLower[i, j]
        
        for i in [1]:
            for j in range(3, 5):
                aUpper[i, j] *= 0.45
                aLower[i, j] *= 0.45
                aUpper[j, i] = aUpper[i, j]
                aLower[j, i] = aLower[i, j]
    
    return aLower, aUpper

def greedy_chooser(env, cands):
    """Greedy MPNN chooser (2023 AccelerateOED)"""
    return choose_next_pair_greedy(env, cands)


def dad_chooser_factory(policy: DADPolicy):
    """DAD policy chooser"""
    def dad_chooser(env, cands):
        hist_tokens = make_hist_tokens(env.h, env.N)
        return policy.choose(env, hist_tokens)
    return dad_chooser


def run_episode_with_metrics(env, chooser_fn, sim_opts):
    """Run episode and compute comprehensive metrics"""
    start_time = time.time()
    
    intermediate_results = []
    
    for step in range(env.K):
        step_start = time.time()
        
        belief_graph = env.features()
        candidates = env.candidate_pairs()
        
        # Compute current MOCU
        current_mocu = None
        if env.surrogate is not None:
            try:
                current_mocu = env.surrogate.forward_mocu(belief_graph).item()
            except:
                pass
        
        # Choose and execute
        xi = chooser_fn(env, candidates)
        result = env.step(xi)
        
        step_time = time.time() - step_start
        
        intermediate_results.append({
            'step': step,
            'chosen_pair': xi,
            'outcome': result['y'],
            'mocu': current_mocu,
            'step_time': step_time
        })
    
    # Compute terminal metrics
    A_min = env.h.lower.copy()
    np.fill_diagonal(A_min, 0.0)
    
    def check_fn(a_ctrl):
        try:
            return sync_check(A_min, env.omega, a_ctrl, **sim_opts)
        except:
            return False
    
    try:
        a_ctrl_star = find_min_a_ctrl(A_min, env.omega, check_fn, 
                                      tol=0.005, max_iter=30, verbose=False)
    except:
        a_ctrl_star = 2.0
    
    # Terminal MOCU
    final_belief_graph = env.features()
    terminal_mocu = None
    if env.surrogate is not None:
        try:
            terminal_mocu = env.surrogate.forward_mocu(final_belief_graph).item()
        except:
            pass
    
    total_time = time.time() - start_time
    
    return {
        'a_ctrl_star': a_ctrl_star,
        'terminal_mocu': terminal_mocu,
        'intermediate_results': intermediate_results,
        'total_time': total_time,
        'A_min': A_min,
        'A_true': env.A_true
    }


def run_strategy_evaluation(env_factory, chooser_fn, sim_opts, n_episodes, 
                           strategy_name):
    """Run evaluation for one strategy with progress bar"""
    results = []
    
    if TQDM_AVAILABLE:
        pbar = tqdm(range(n_episodes), desc=f"Evaluating {strategy_name}", 
                   unit="ep", ncols=100,
                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        iterator = pbar
    else:
        print(f"\nEvaluating {strategy_name}: {n_episodes} episodes")
        iterator = range(n_episodes)
    
    for ep_idx in iterator:
        env = env_factory()
        result = run_episode_with_metrics(env, chooser_fn, sim_opts)
        results.append(result)
        
        if not TQDM_AVAILABLE and (ep_idx + 1) % 10 == 0:
            print(f"  {ep_idx + 1}/{n_episodes} complete")
    
    if TQDM_AVAILABLE:
        pbar.close()
    
    return results


def compute_statistics(results: List[Dict]) -> Dict[str, Any]:
    """Compute comprehensive statistics"""
    a_ctrl_stars = [r['a_ctrl_star'] for r in results]
    terminal_mocus = [r['terminal_mocu'] for r in results if r['terminal_mocu'] is not None]
    total_times = [r['total_time'] for r in results]
    
    stats = {
        'n_episodes': len(results),
        'a_ctrl_star': {
            'mean': float(np.mean(a_ctrl_stars)),
            'std': float(np.std(a_ctrl_stars)),
            'min': float(np.min(a_ctrl_stars)),
            'max': float(np.max(a_ctrl_stars)),
            'median': float(np.median(a_ctrl_stars))
        },
        'terminal_mocu': None,
        'total_time': {
            'mean': float(np.mean(total_times)),
            'std': float(np.std(total_times)),
            'min': float(np.min(total_times)),
            'max': float(np.max(total_times))
        }
    }
    
    if terminal_mocus:
        stats['terminal_mocu'] = {
            'mean': float(np.mean(terminal_mocus)),
            'std': float(np.std(terminal_mocus)),
            'min': float(np.min(terminal_mocus)),
            'max': float(np.max(terminal_mocus)),
            'median': float(np.median(terminal_mocus))
        }
    
    return stats


def load_models(models_dir: str, cfg: Dict, device: str) -> Dict:
    """Load all available models"""
    print("\n" + "="*80)
    print("LOADING MODELS")
    print("="*80)
    
    models = {}
    
    # Load surrogate
    surrogate_path = os.path.join(models_dir, "mpnn_surrogate.pth")
    if os.path.exists(surrogate_path):
        print(f"Loading: mpnn_surrogate.pth")
        try:
            surrogate = MPNNSurrogate(
                mocu_scale=cfg["surrogate"].get("mocu_scale", 1.0),
                hidden=cfg["surrogate"]["hidden"],
                dropout=cfg["surrogate"]["dropout"]
            )
            surrogate.load_state_dict(torch.load(surrogate_path, map_location=device, 
                                                weights_only=True))
            surrogate.to(device)
            surrogate.eval()
            models["surrogate"] = surrogate
            print("  SUCCESS")
        except Exception as e:
            print(f"  ERROR: {e}")
    else:
        print(f"Not found: mpnn_surrogate.pth")
    
    # Load fixed design
    fixed_path = os.path.join(models_dir, "fixed_design.pkl")
    if os.path.exists(fixed_path):
        print(f"Loading: fixed_design.pkl")
        with open(fixed_path, 'rb') as f:
            models["fixed_design"] = pickle.load(f)
        print(f"  SUCCESS: {models['fixed_design']}")
    else:
        print(f"Not found: fixed_design.pkl")
    
    # Load DAD policy
    dad_path = os.path.join(models_dir, "dad_policy.pth")
    if os.path.exists(dad_path):
        print(f"Loading: dad_policy.pth")
        try:
            policy = DADPolicy(hidden=cfg["dad_rl"]["hidden"])
            policy.load_state_dict(torch.load(dad_path, map_location='cpu', 
                                             weights_only=True))
            policy.eval()
            models["dad_policy"] = policy
            print("  SUCCESS")
        except Exception as e:
            print(f"  ERROR: {e}")
    else:
        print(f"Not found: dad_policy.pth")
    
    print("="*80)
    return models


def build_strategies(models: Dict) -> Dict:
    """Build strategy dictionary"""
    strategies = {}
    
    # Always available
    strategies["Random"] = random_chooser
    
    # Requires surrogate
    if "surrogate" in models:
        strategies["Greedy MPNN (2023)"] = greedy_chooser
    
    # Requires fixed design
    if "fixed_design" in models:
        strategies["Fixed Design (Static)"] = fixed_design_chooser_factory(
            models["fixed_design"]
        )
    
    # Requires DAD policy
    if "dad_policy" in models:
        strategies["DAD (MOCU-based)"] = dad_chooser_factory(models["dad_policy"])
    
    return strategies


def print_results(comparison_results: Dict):
    """Print formatted results"""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    for strategy_name, data in comparison_results.items():
        stats = data['statistics']
        print(f"\n{strategy_name}:")
        print("-" * len(strategy_name))
        
        # a_ctrl_star
        a_ctrl = stats['a_ctrl_star']
        print(f"  a_ctrl_star: {a_ctrl['mean']:.4f} +/- {a_ctrl['std']:.4f}")
        print(f"    Range: [{a_ctrl['min']:.4f}, {a_ctrl['max']:.4f}]")
        print(f"    Median: {a_ctrl['median']:.4f}")
        
        # terminal MOCU
        if stats['terminal_mocu'] is not None:
            mocu = stats['terminal_mocu']
            print(f"  Terminal MOCU: {mocu['mean']:.4f} +/- {mocu['std']:.4f}")
            print(f"    Range: [{mocu['min']:.4f}, {mocu['max']:.4f}]")
        
        # Time
        t = stats['total_time']
        print(f"  Time per episode: {t['mean']:.2f}s")
        print(f"  Total episodes: {stats['n_episodes']}")


def compute_improvements(comparison_results: Dict) -> Dict:
    """Compute improvements vs Random baseline"""
    if "Random" not in comparison_results:
        return {}
    
    baseline = comparison_results["Random"]['statistics']
    improvements = {}
    
    for strategy_name, data in comparison_results.items():
        if strategy_name == "Random":
            continue
        
        stats = data['statistics']
        
        # a_ctrl improvement (lower is better)
        a_ctrl_improv = ((baseline['a_ctrl_star']['mean'] - 
                         stats['a_ctrl_star']['mean']) / 
                        baseline['a_ctrl_star']['mean'] * 100)
        
        improvement_data = {
            'a_ctrl_improvement_pct': a_ctrl_improv
        }
        
        # MOCU improvement (lower is better)
        if (stats['terminal_mocu'] is not None and 
            baseline['terminal_mocu'] is not None):
            mocu_improv = ((baseline['terminal_mocu']['mean'] - 
                          stats['terminal_mocu']['mean']) / 
                         baseline['terminal_mocu']['mean'] * 100)
            improvement_data['mocu_improvement_pct'] = mocu_improv
        
        improvements[strategy_name] = improvement_data
    
    return improvements


def save_results_json(comparison_results: Dict, filepath: str):
    """Save results to JSON"""
    def convert(obj):
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        else:
            return obj
    
    with open(filepath, 'w') as f:
        json.dump(convert(comparison_results), f, indent=2)
    print(f"\nResults saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all design strategies"
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--cpu", action="store_true")
    
    args = parser.parse_args()
    
    # Setup
    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    cfg = load_config(args.config)
    
    print("="*80)
    print("COMPREHENSIVE EVALUATION")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Episodes: {args.episodes}")
    print(f"Device: {device.upper()}")
    print("="*80)
    
    # Load models
    models = load_models(args.models_dir, cfg, device)
    
    if not models:
        print("\nERROR: No models found")
        print("Run: python train.py --mode dad_with_surrogate")
        return 1
    
    # Build strategies
    strategies = build_strategies(models)
    
    print("\n" + "="*80)
    print("STRATEGIES TO EVALUATE")
    print("="*80)
    for i, name in enumerate(strategies.keys(), 1):
        print(f"{i}. {name}")
    print("="*80)
    
    # Create environment factory
    surrogate = models.get("surrogate", None)
    env_factory = make_env_factory(cfg, surrogate=surrogate)
    sim_opts = cfg["sim"]
    
    # Run evaluations
    comparison_results = {}
    
    for strategy_name, chooser_fn in strategies.items():
        results = run_strategy_evaluation(
            env_factory, chooser_fn, sim_opts, 
            args.episodes, strategy_name
        )
        stats = compute_statistics(results)
        comparison_results[strategy_name] = {
            'results': results,
            'statistics': stats
        }
    
    # Print results
    print_results(comparison_results)
    
    # Compute improvements
    improvements = compute_improvements(comparison_results)
    if improvements:
        print("\n" + "="*80)
        print("IMPROVEMENT vs Random Baseline")
        print("="*80)
        for strategy, metrics in improvements.items():
            print(f"\n{strategy}:")
            if 'mocu_improvement_pct' in metrics:
                print(f"  MOCU: {metrics['mocu_improvement_pct']:+.1f}%")
            print(f"  a_ctrl: {metrics['a_ctrl_improvement_pct']:+.1f}%")
    
    # Save results
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    save_results_json(comparison_results, results_path)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        exit(1)
