import yaml
import numpy as np
import torch
import os
import argparse
from typing import Dict, Callable, Any

from core.kuramoto_env import PairTestEnv
from surrogate.mpnn_surrogate import MPNNSurrogate
from surrogate.train_surrogate import train_surrogate_model
from design.greedy_erm import choose_next_pair_greedy
from design.dad_policy import DADPolicy
from design.train_bc import train_behavior_cloning
from eval.run_eval import run_episode, run_multiple_episodes, compare_strategies
from eval.metrics import print_comparison_results, plot_mocu_curves, plot_a_ctrl_distribution, save_results
from data_generation.synthetic_data import SyntheticDataGenerator


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
        return PairTestEnv(N=N, omega=omega, prior_bounds=prior, K=K, surrogate=surrogate, rng=rng)
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
        model = MPNNSurrogate(mocu_scale=cfg["surrogate"]["mocu_scale"])
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    
    print("Training surrogate model...")
    model, _ = train_surrogate_model(
        N=cfg["N"], K=cfg["K"], 
        n_train=cfg.get("surrogate", {}).get("n_train", 1000),
        n_val=cfg.get("surrogate", {}).get("n_val", 200),
        epochs=cfg.get("surrogate", {}).get("epochs", 50),
        device='cpu',
        save_path=model_path
    )
    return model


def train_dad_policy(cfg: Dict[str, Any], surrogate: MPNNSurrogate) -> DADPolicy:
    """Train DAD policy using behavior cloning."""
    print("Training DAD policy...")
    
    # Create environment factory for training
    env_factory = make_env_factory(cfg, surrogate=surrogate)
    
    # Initialize policy
    policy = DADPolicy(hidden=64)
    
    # Train using behavior cloning
    bc_config = cfg.get("dad_bc", {})
    trained_policy = train_behavior_cloning(
        env_factory=env_factory,
        policy=policy,
        epochs=bc_config.get("epochs", 3),
        episodes_per_epoch=bc_config.get("episodes_per_epoch", 30),
        lr=bc_config.get("lr", 1e-3)
    )
    
    return trained_policy


def run_comprehensive_evaluation(cfg: Dict[str, Any], n_episodes: int = 10, 
                                verbose: bool = False, save_results_path: str = None):
    """Run comprehensive evaluation comparing all strategies."""
    print("="*80)
    print("COMPREHENSIVE EVALUATION")
    print("="*80)
    
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
            print("Continuing with Random and GreedyERM only...")
    
    # Run comparison
    print(f"\nRunning evaluation with {n_episodes} episodes per strategy...")
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
    
    # Generate plots
    try:
        plot_mocu_curves(comparison_results, save_path="mocu_curves.png")
        plot_a_ctrl_distribution(comparison_results, save_path="a_ctrl_distribution.png")
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
    
    # Save results
    if save_results_path:
        save_results(comparison_results, save_results_path)
    
    return comparison_results


def run_quick_demo(cfg: Dict[str, Any]):
    """Run a quick demo with single episodes."""
    print("="*80)
    print("QUICK DEMO")
    print("="*80)
    
    # Load or train surrogate
    surrogate = train_surrogate_if_needed(cfg)
    sim = cfg["sim"]
    
    # Test strategies
    strategies = {
        "GreedyERM": greedy_chooser,
        "Random": random_chooser,
    }
    
    for name, chooser in strategies.items():
        print(f"\nTesting {name}...")
        env_factory = make_env_factory(cfg, surrogate=surrogate, seed=0 if name=="GreedyERM" else 1)
        env = env_factory()
        out = run_episode(env, chooser, sim_opts=sim, verbose=True)
        print(f"[{name}] a_ctrl* = {out['a_ctrl_star']:.4f}, terminal MOCU_hat = {out['terminal_mocu_hat']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Kuramoto Experiment Design Demo")
    parser.add_argument("--config", default="configs/exp_fixedK.yaml", help="Config file path")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick", help="Demo mode")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes for full evaluation")
    parser.add_argument("--retrain", action="store_true", help="Force retrain surrogate model")
    parser.add_argument("--save-results", help="Path to save results JSON")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_cfg(args.config)
    
    if args.mode == "quick":
        run_quick_demo(cfg)
    else:
        run_comprehensive_evaluation(
            cfg, 
            n_episodes=args.episodes,
            verbose=args.verbose,
            save_results_path=args.save_results
        )


if __name__ == "__main__":
    main()
