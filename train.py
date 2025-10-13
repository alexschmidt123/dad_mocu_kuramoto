#!/usr/bin/env python3
"""
Training script for Kuramoto experiment design methods.

Trains models in stages:
1. MPNN Surrogate (required for all adaptive methods)
2. Fixed Design (generates static sequence using MPNN)
3. DAD Policy (learns adaptive policy on top of MPNN)

Usage:
  # Train all methods
  python train.py --config configs/config.yaml --methods all
  
  # Train specific methods
  python train.py --methods surrogate dad
  python train.py --methods surrogate fixed
  
  # Quick training
  python train.py --config configs/config_fast.yaml --methods all
  
  # Force retrain
  python train.py --methods all --force
"""

import yaml
import numpy as np
import torch
import argparse
import os
from typing import Dict, Any, List

from core.kuramoto_env import PairTestEnv
from surrogate.mpnn_surrogate import MPNNSurrogate
from surrogate.train_surrogate import train_surrogate_model
from design.greedy_erm import choose_next_pair_greedy
from design.dad_policy import DADPolicy
from design.train_bc import train_behavior_cloning


def setup_gpu():
    """Configure GPU for optimal performance."""
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available")
        return False
    
    torch.backends.cudnn.benchmark = True
    
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
    print("="*80)
    return True


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_env_factory(cfg: Dict, surrogate):
    """Create environment factory."""
    def env_factory():
        N, K = cfg["N"], cfg["K"]
        rng = np.random.default_rng(cfg["seed"])
        
        if cfg["omega"]["kind"] == "uniform":
            omega = rng.uniform(cfg["omega"]["low"], cfg["omega"]["high"], size=N)
        else:
            omega = rng.normal(0.0, 1.0, size=N)
        
        prior = (cfg["prior_lower"], cfg["prior_upper"])
        return PairTestEnv(N=N, omega=omega, prior_bounds=prior, K=K, 
                          surrogate=surrogate, rng=rng)
    return env_factory


def train_surrogate(cfg: Dict, device: str, models_dir: str, force: bool = False) -> MPNNSurrogate:
    """Train MPNN surrogate model."""
    model_path = os.path.join(models_dir, "mpnn_surrogate.pth")
    
    print("\n" + "="*80)
    print("TRAINING: MPNN Surrogate Model")
    print("="*80)
    
    if not force and os.path.exists(model_path):
        print(f"⊙ Loading existing model from {model_path}")
        surrogate = MPNNSurrogate(
            mocu_scale=cfg["surrogate"].get("mocu_scale", 1.0),
            hidden=cfg["surrogate"]["hidden"],
            dropout=cfg["surrogate"]["dropout"]
        )
        surrogate.load_state_dict(torch.load(model_path, map_location=device))
        surrogate.to(device)
        surrogate.eval()
        print("✓ Model loaded\n")
        return surrogate
    
    print("Training new surrogate model...")
    surrogate, results = train_surrogate_model(
        N=cfg["N"],
        K=cfg["K"],
        n_train=cfg["surrogate"]["n_train"],
        n_val=cfg["surrogate"]["n_val"],
        n_theta_samples=cfg["surrogate"]["n_theta_samples"],
        epochs=cfg["surrogate"]["epochs"],
        lr=cfg["surrogate"]["lr"],
        batch_size=cfg["surrogate"]["batch_size"],
        device=device,
        save_path=model_path,
        use_cache=True
    )
    
    print(f"✓ Surrogate model saved to {model_path}")
    print(f"  Final MOCU loss: {results['mocu']['train_losses'][-1]:.4f}")
    print(f"  Final ERM loss: {results['erm']['train_losses'][-1]:.4f}")
    print()
    
    return surrogate


def train_fixed_design(cfg: Dict, surrogate: MPNNSurrogate, models_dir: str, 
                       force: bool = False) -> List:
    """Generate fixed design sequence using greedy MPNN."""
    import pickle
    
    model_path = os.path.join(models_dir, "fixed_design.pkl")
    
    print("\n" + "="*80)
    print("TRAINING: Fixed Design (Static Sequence)")
    print("="*80)
    
    if not force and os.path.exists(model_path):
        print(f"⊙ Loading existing design from {model_path}")
        with open(model_path, 'rb') as f:
            fixed_sequence = pickle.load(f)
        print(f"✓ Fixed sequence: {fixed_sequence}\n")
        return fixed_sequence
    
    print("Generating fixed design sequence using Greedy MPNN...")
    env_factory = make_env_factory(cfg, surrogate=surrogate)
    representative_env = env_factory()
    fixed_sequence = []
    
    for step in range(cfg["K"]):
        cands = representative_env.candidate_pairs()
        xi = choose_next_pair_greedy(representative_env, cands)
        fixed_sequence.append(xi)
        representative_env.step(xi)
        print(f"  Step {step+1}: Selected pair {xi}")
    
    # Save
    with open(model_path, 'wb') as f:
        pickle.dump(fixed_sequence, f)
    
    print(f"✓ Fixed design saved to {model_path}")
    print(f"  Sequence: {fixed_sequence}\n")
    
    return fixed_sequence


def train_dad_policy(cfg: Dict, surrogate: MPNNSurrogate, models_dir: str,
                     force: bool = False) -> DADPolicy:
    """Train DAD policy via behavior cloning."""
    model_path = os.path.join(models_dir, "dad_policy.pth")
    
    print("\n" + "="*80)
    print("TRAINING: DAD Policy (Deep Adaptive Design)")
    print("="*80)
    
    if not force and os.path.exists(model_path):
        print(f"⊙ Loading existing policy from {model_path}")
        policy = DADPolicy(hidden=cfg["dad_bc"]["hidden"])
        policy.load_state_dict(torch.load(model_path, map_location='cpu'))
        policy.eval()
        print("✓ Policy loaded\n")
        return policy
    
    print("Training DAD policy via behavior cloning on Greedy MPNN trajectories...")
    env_factory = make_env_factory(cfg, surrogate=surrogate)
    policy = DADPolicy(hidden=cfg["dad_bc"]["hidden"])
    
    policy = train_behavior_cloning(
        env_factory=env_factory,
        policy=policy,
        epochs=cfg["dad_bc"]["epochs"],
        episodes_per_epoch=cfg["dad_bc"]["episodes_per_epoch"],
        lr=cfg["dad_bc"]["lr"]
    )
    
    # Save
    torch.save(policy.state_dict(), model_path)
    print(f"✓ DAD policy saved to {model_path}\n")
    
    return policy


def main():
    parser = argparse.ArgumentParser(
        description="Train models for Kuramoto experiment design",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training Methods:
  surrogate  - MPNN surrogate (predicts MOCU/ERM/Sync) [required for all]
  fixed      - Fixed design (static sequence using MPNN)
  dad        - DAD policy (adaptive policy using MPNN)
  all        - Train all methods

Examples:
  # Train everything
  python train.py --methods all
  
  # Train only surrogate
  python train.py --methods surrogate
  
  # Train surrogate and DAD
  python train.py --methods surrogate dad
  
  # Quick training
  python train.py --config configs/config_fast.yaml --methods all
  
  # Force retrain everything
  python train.py --methods all --force
        """
    )
    parser.add_argument("--config", default="configs/config.yaml",
                       help="Config file")
    parser.add_argument("--methods", nargs="+", 
                       choices=["surrogate", "fixed", "dad", "all"],
                       default=["all"],
                       help="Methods to train")
    parser.add_argument("--force", action="store_true",
                       help="Force retrain (ignore existing models)")
    parser.add_argument("--models-dir", default="models",
                       help="Directory to save trained models")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU mode")
    
    args = parser.parse_args()
    
    # Expand 'all'
    if "all" in args.methods:
        args.methods = ["surrogate", "fixed", "dad"]
    
    # Setup
    if not args.cpu:
        gpu_ok = setup_gpu()
    else:
        gpu_ok = False
        print("Running in CPU mode")
    
    device = 'cuda' if gpu_ok else 'cpu'
    
    # Create models directory
    os.makedirs(args.models_dir, exist_ok=True)
    
    # Load config
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found!")
        return 1
    
    cfg = load_config(args.config)
    
    print("\n" + "="*80)
    print("TRAINING PIPELINE")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Methods: {args.methods}")
    print(f"Models dir: {args.models_dir}")
    print(f"Device: {device.upper()}")
    print(f"Force retrain: {args.force}")
    print("="*80)
    
    trained = {}
    
    try:
        # Stage 1: MPNN Surrogate (required for all methods)
        if "surrogate" in args.methods or "fixed" in args.methods or "dad" in args.methods:
            surrogate = train_surrogate(cfg, device, args.models_dir, args.force)
            trained["surrogate"] = True
        else:
            # Load existing surrogate if needed
            model_path = os.path.join(args.models_dir, "mpnn_surrogate.pth")
            if os.path.exists(model_path):
                print(f"⊙ Loading surrogate from {model_path}")
                surrogate = MPNNSurrogate(
                    mocu_scale=cfg["surrogate"].get("mocu_scale", 1.0),
                    hidden=cfg["surrogate"]["hidden"],
                    dropout=cfg["surrogate"]["dropout"]
                )
                surrogate.load_state_dict(torch.load(model_path, map_location=device))
                surrogate.to(device)
                surrogate.eval()
            else:
                print("Error: Surrogate model not found. Train it first:")
                print("  python train.py --methods surrogate")
                return 1
        
        # Stage 2: Fixed Design
        if "fixed" in args.methods:
            fixed_sequence = train_fixed_design(cfg, surrogate, args.models_dir, args.force)
            trained["fixed"] = True
        
        # Stage 3: DAD Policy
        if "dad" in args.methods:
            dad_policy = train_dad_policy(cfg, surrogate, args.models_dir, args.force)
            trained["dad"] = True
        
        # Summary
        print("\n" + "="*80)
        print("✓ TRAINING COMPLETE")
        print("="*80)
        print(f"\nTrained methods: {list(trained.keys())}")
        print(f"Models saved in: {args.models_dir}/")
        print("\nFiles:")
        for filename in os.listdir(args.models_dir):
            filepath = os.path.join(args.models_dir, filename)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  {filename} ({size_mb:.2f} MB)")
        
        print("\nNext steps:")
        print(f"  python test.py --config {args.config} --models-dir {args.models_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\n\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        exit(1)