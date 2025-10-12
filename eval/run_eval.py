from core.bisection import find_min_a_ctrl
from core.pacemaker_control import sync_check
import numpy as np
from typing import Dict, List, Callable, Any
import time


def run_episode(env, chooser_fn, sim_opts, verbose=False):
    """Run a single episode of the experiment design process."""
    start_time = time.time()
    
    # Store intermediate results
    intermediate_results = []
    
    for step in range(env.K):
        step_start = time.time()
        
        # Get current belief state
        belief_graph = env.features()
        
        # Choose next pair
        candidates = env.candidate_pairs()
        xi = chooser_fn(env, candidates)
        
        # Run experiment
        result = env.step(xi)
        
        # Compute current MOCU if surrogate is available
        current_mocu = None
        if env.surrogate is not None:
            current_mocu = env.surrogate.forward_mocu(belief_graph).item()
        
        step_time = time.time() - step_start
        
        intermediate_results.append({
            'step': step,
            'chosen_pair': xi,
            'outcome': result['y'],
            'mocu': current_mocu,
            'step_time': step_time,
            'belief_graph': belief_graph
        })
        
        if verbose:
            print(f"Step {step+1}/{env.K}: Pair {xi}, Outcome: {result['y']}, "
                  f"MOCU: {current_mocu:.4f if current_mocu else 'N/A'}")
    
    # Compute final control parameter
    A_min = env.h.lower.copy()
    check = lambda a: sync_check(A_min, env.omega, a, **sim_opts)
    a_ctrl_star = find_min_a_ctrl(A_min, env.omega, check_fn=check)
    
    # Final MOCU
    final_belief_graph = env.features()
    term_mocu = env.surrogate.forward_mocu(final_belief_graph).item() if env.surrogate is not None else None
    
    total_time = time.time() - start_time
    
    return {
        'a_ctrl_star': a_ctrl_star,
        'terminal_mocu_hat': term_mocu,
        'intermediate_results': intermediate_results,
        'total_time': total_time,
        'A_min': A_min,
        'final_belief_graph': final_belief_graph
    }


def run_multiple_episodes(env_factory, chooser_fn, sim_opts, n_episodes=10, 
                         verbose=False, seed=None):
    """Run multiple episodes and collect statistics."""
    if seed is not None:
        np.random.seed(seed)
    
    results = []
    
    for episode in range(n_episodes):
        if verbose:
            print(f"\nRunning episode {episode+1}/{n_episodes}")
        
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


def compare_strategies(env_factory, strategies: Dict[str, Callable], sim_opts, 
                     n_episodes=10, verbose=False, seed=None):
    """Compare multiple strategies on the same problem instances."""
    if seed is not None:
        np.random.seed(seed)
    
    comparison_results = {}
    
    for strategy_name, chooser_fn in strategies.items():
        if verbose:
            print(f"\nEvaluating strategy: {strategy_name}")
        
        results = run_multiple_episodes(env_factory, chooser_fn, sim_opts, 
                                      n_episodes, verbose=False, seed=None)
        stats = compute_statistics(results)
        comparison_results[strategy_name] = {
            'results': results,
            'statistics': stats
        }
    
    return comparison_results
