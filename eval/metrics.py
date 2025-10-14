import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import json


def summary(metrics: dict) -> str:
    """Create a summary string from metrics dictionary."""
    return "\n".join(f"{k}: {v}" for k,v in metrics.items())


def print_comparison_results(comparison_results: Dict[str, Dict], 
                           metrics: List[str] = ['a_ctrl_star', 'terminal_mocu']):
    """Print formatted comparison results."""
    print("\n" + "="*80)
    print("STRATEGY COMPARISON RESULTS")
    print("="*80)
    
    for strategy_name, data in comparison_results.items():
        print(f"\n{strategy_name}:")
        print("-" * len(strategy_name))
        
        stats = data['statistics']
        for metric in metrics:
            if metric in stats and stats[metric] is not None:
                m = stats[metric]
                print(f"  {metric}:")
                print(f"    Mean: {m['mean']:.4f} ± {m['std']:.4f}")
                print(f"    Range: [{m['min']:.4f}, {m['max']:.4f}]")
                print(f"    Median: {m['median']:.4f}")
            else:
                print(f"  {metric}: N/A")
        
        print(f"  Episodes: {stats['n_episodes']}")
        print(f"  Avg Time: {stats['total_time']['mean']:.4f}s")


def plot_mocu_curves(comparison_results: Dict[str, Dict], save_path: str = None):
    """Plot MOCU curves for different strategies."""
    plt.figure(figsize=(10, 6))
    
    for strategy_name, data in comparison_results.items():
        results = data['results']
        
        # Extract MOCU values over time
        mocu_curves = []
        for result in results:
            curve = []
            for step_data in result['intermediate_results']:
                if step_data['mocu'] is not None:
                    curve.append(step_data['mocu'])
            if curve:
                mocu_curves.append(curve)
        
        if mocu_curves:
            # Pad curves to same length
            max_len = max(len(curve) for curve in mocu_curves)
            padded_curves = []
            for curve in mocu_curves:
                padded = curve + [curve[-1]] * (max_len - len(curve))
                padded_curves.append(padded)
            
            # Compute mean and std
            curves_array = np.array(padded_curves)
            mean_curve = np.mean(curves_array, axis=0)
            std_curve = np.std(curves_array, axis=0)
            
            steps = np.arange(1, len(mean_curve) + 1)  # Start from 1, not 0
            plt.plot(steps, mean_curve, label=strategy_name, linewidth=2)
            plt.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve, alpha=0.3)
    
    plt.xlabel('Experiment Step')
    plt.ylabel('MOCU')
    plt.title('MOCU Evolution Over Experiment Steps')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_a_ctrl_distribution(comparison_results: Dict[str, Dict], save_path: str = None):
    """Plot distribution of final control parameters."""
    plt.figure(figsize=(10, 6))
    
    a_ctrl_data = {}
    for strategy_name, data in comparison_results.items():
        a_ctrl_stars = [r['a_ctrl_star'] for r in data['results']]
        a_ctrl_data[strategy_name] = a_ctrl_stars
    
    plt.hist([a_ctrl_data[name] for name in a_ctrl_data.keys()], 
             bins=20, alpha=0.7, label=list(a_ctrl_data.keys()))
    
    plt.xlabel('Final Control Parameter (a_ctrl*)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Final Control Parameters')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def save_results(comparison_results: Dict[str, Dict], filepath: str):
    """Save comparison results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_numpy(comparison_results)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {filepath}")


def load_results(filepath: str) -> Dict[str, Dict]:
    """Load comparison results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def compute_improvement_metrics(comparison_results: Dict[str, Dict], 
                              baseline_strategy: str = 'Random') -> Dict[str, Dict]:
    """Compute improvement metrics relative to baseline strategy."""
    if baseline_strategy not in comparison_results:
        raise ValueError(f"Baseline strategy '{baseline_strategy}' not found in results")
    
    baseline_stats = comparison_results[baseline_strategy]['statistics']
    improvements = {}
    
    for strategy_name, data in comparison_results.items():
        if strategy_name == baseline_strategy:
            continue
        
        stats = data['statistics']
        improvement = {}
        
        # MOCU improvement (lower is better)
        if (stats['terminal_mocu'] is not None and 
            baseline_stats['terminal_mocu'] is not None):
            mocu_improvement = ((baseline_stats['terminal_mocu']['mean'] - 
                               stats['terminal_mocu']['mean']) / 
                              baseline_stats['terminal_mocu']['mean'] * 100)
            improvement['mocu_improvement_pct'] = mocu_improvement
        
        # Control parameter improvement (lower is better)
        a_ctrl_improvement = ((baseline_stats['a_ctrl_star']['mean'] - 
                             stats['a_ctrl_star']['mean']) / 
                            baseline_stats['a_ctrl_star']['mean'] * 100)
        improvement['a_ctrl_improvement_pct'] = a_ctrl_improvement
        
        improvements[strategy_name] = improvement
    
    return improvements
