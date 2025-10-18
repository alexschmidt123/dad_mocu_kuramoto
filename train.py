#!/usr/bin/env python3
"""
Training pipeline aligned with AccelerateOED 2023 and DAD papers
Supports: surrogate-only and dad_with_surrogate modes
"""

import yaml
import numpy as np
import torch
import argparse
import os
import pickle
from typing import Dict, Any

from core.kuramoto_env import PairTestEnv
from surrogate.mpnn_surrogate import MPNNSurrogate
from surrogate.train_surrogate import train_surrogate_model
from design.greedy_erm import choose_next_pair_greedy
from design.dad_policy import DADPolicy
from design.train_rl import train_dad_rl

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def setup_gpu():
    """Configure GPU"""
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        return False
    
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        torch.backends.cuda.matmul.allow_tf32 = True
    
    print("="*80)
    print("GPU Configuration")
    print("="*80)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print("="*80)
    return True


def load_config(path: str) -> Dict[str, Any]:
    """Load config"""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_env_factory(cfg: Dict, surrogate):
    """Create environment factory"""
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


def train_surrogate_mode(cfg: Dict, device: str, models_dir: str, force: bool):
    """
    Train MPNN surrogate following AccelerateOED 2023 approach.
    
    Key alignments with 2023 paper:
    - Graph-based inputs (node/edge features)
    - MPNN message passing layers
    - Target normalization
    - Adam optimizer with lr=0.001
    - Batch training
    """
    model_path = os.path.join(models_dir, "mpnn_surrogate.pth")
    
    print("\n" + "="*80)
    print("SURROGATE TRAINING (AccelerateOED 2023 aligned)")
    print("="*80)
    
    if not force and os.path.exists(model_path):
        print(f"Model exists: {model_path}")
        print("Use --force to retrain")
        try:
            surrogate = MPNNSurrogate(
                mocu_scale=cfg["surrogate"].get("mocu_scale", 1.0),
                hidden=cfg["surrogate"]["hidden"],
                dropout=cfg["surrogate"]["dropout"]
            )
            surrogate.load_state_dict(torch.load(model_path, map_location=device, 
                                                weights_only=True))
            surrogate.to(device)
            surrogate.eval()
            print("Model loaded successfully")
            return surrogate
        except Exception as e:
            print(f"WARNING: Failed to load: {e}")
            print("Training new model...")
    
    print("\nTraining Configuration:")
    print(f"  Architecture: MPNN with {cfg['surrogate']['hidden']} hidden dims")
    print(f"  Message passing layers: 3")
    print(f"  Training samples: {cfg['surrogate']['n_train']}")
    print(f"  Validation samples: {cfg['surrogate']['n_val']}")
    print(f"  Epochs: {cfg['surrogate']['epochs']}")
    print(f"  Learning rate: {cfg['surrogate']['lr']}")
    print(f"  Batch size: {cfg['surrogate']['batch_size']}")
    print(f"  Optimizer: Adam")
    print(f"  Target normalization: Enabled")
    print("="*80)
    
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
        use_cache=True,
        hidden=cfg["surrogate"]["hidden"],
        dropout=cfg["surrogate"]["dropout"],
        mocu_scale=cfg["surrogate"].get("mocu_scale", 1.0)
    )
    
    print("\n" + "="*80)
    print("SURROGATE TRAINING COMPLETE")
    print("="*80)
    print(f"Model saved: {model_path}")
    print(f"Final MOCU loss: {results['mocu']['train_losses'][-1]:.4f}")
    if results['erm']['train_losses']:
        print(f"Final ERM loss: {results['erm']['train_losses'][-1]:.4f}")
    print("="*80)
    
    return surrogate


def train_fixed_design(cfg: Dict, surrogate: MPNNSurrogate, models_dir: str, force: bool):
    """
    Generate fixed design using greedy MPNN selection.
    This is the 2023 AccelerateOED baseline design.
    """
    model_path = os.path.join(models_dir, "fixed_design.pkl")
    
    print("\n" + "="*80)
    print("FIXED DESIGN (2023 AccelerateOED Baseline)")
    print("="*80)
    
    if not force and os.path.exists(model_path):
        print(f"Design exists: {model_path}")
        with open(model_path, 'rb') as f:
            fixed_sequence = pickle.load(f)
        print(f"Sequence: {fixed_sequence}")
        return fixed_sequence
    
    print("Generating fixed sequence using Greedy MPNN...")
    print("This design is computed ONCE offline and reused for all instances")
    
    env_factory = make_env_factory(cfg, surrogate=surrogate)
    env = env_factory()
    fixed_sequence = []
    
    for step in range(cfg["K"]):
        all_cands = env.candidate_pairs()
        untested_cands = [(i, j) for (i, j) in all_cands if not env.h.tested[i, j]]
        cands = untested_cands if untested_cands else all_cands
        
        xi = choose_next_pair_greedy(env, cands)
        fixed_sequence.append(xi)
        env.step(xi)
        
        print(f"  Step {step+1}: {xi}")
    
    with open(model_path, 'wb') as f:
        pickle.dump(fixed_sequence, f)
    
    print(f"Saved: {model_path}")
    print(f"Fixed sequence: {fixed_sequence}")
    return fixed_sequence


def train_dad_policy_mode(cfg: Dict, surrogate: MPNNSurrogate, models_dir: str, force: bool):
    """
    Train DAD policy adapted for MOCU-based objective.
    
    Key differences from DAD paper (eigenvalue-based):
    1. Objective: Minimize MOCU instead of maximize EIG
    2. Reward: -MOCU (negative because we minimize)
    3. Surrogate: Uses MPNN for MOCU/ERM prediction
    4. Training: REINFORCE policy gradient
    """
    model_path = os.path.join(models_dir, "dad_policy.pth")
    
    print("\n" + "="*80)
    print("DAD POLICY TRAINING (MOCU-based Adaptation)")
    print("="*80)
    
    if not force and os.path.exists(model_path):
        print(f"Policy exists: {model_path}")
        try:
            policy = DADPolicy(hidden=cfg["dad_rl"]["hidden"])
            policy.load_state_dict(torch.load(model_path, map_location='cpu', 
                                             weights_only=True))
            policy.eval()
            print("Policy loaded successfully")
            return policy
        except Exception as e:
            print(f"WARNING: Failed to load: {e}")
            print("Training new policy...")
    
    print("\nDifferences from DAD paper:")
    print("  Original DAD: Eigenvalue-based, maximize EIG")
    print("  Our version: MOCU-based, minimize terminal MOCU")
    print("\nTraining Configuration:")
    print(f"  Policy hidden dims: {cfg['dad_rl']['hidden']}")
    print(f"  Epochs: {cfg['dad_rl']['epochs']}")
    print(f"  Episodes/epoch: {cfg['dad_rl']['episodes_per_epoch']}")
    print(f"  Learning rate: {cfg['dad_rl']['lr']}")
    print(f"  Algorithm: REINFORCE")
    print(f"  Reward: -MOCU (lower is better)")
    print("="*80)
    
    env_factory = make_env_factory(cfg, surrogate=surrogate)
    policy = DADPolicy(hidden=cfg["dad_rl"]["hidden"])
    
    policy = train_dad_rl(
        env_factory=env_factory,
        policy=policy,
        surrogate=surrogate,
        epochs=cfg["dad_rl"]["epochs"],
        episodes_per_epoch=cfg["dad_rl"]["episodes_per_epoch"],
        lr=cfg["dad_rl"]["lr"]
    )
    
    torch.save(policy.state_dict(), model_path)
    
    print("\n" + "="*80)
    print("DAD POLICY TRAINING COMPLETE")
    print("="*80)
    print(f"Model saved: {model_path}")
    print("="*80)
    
    return policy


def main():
    parser = argparse.ArgumentParser(
        description="Train models for Kuramoto experiment design",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training Modes:
  surrogate-only: Train MPNN surrogate (2023 AccelerateOED aligned)
  dad_with_surrogate: Train surrogate + fixed design + DAD policy

Examples:
  # Train surrogate only
  python train.py --mode surrogate-only --config configs/config.yaml
  
  # Train all methods (surrogate + designs)
  python train.py --mode dad_with_surrogate --config configs/config.yaml
  
  # Force retrain everything
  python train.py --mode dad_with_surrogate --config configs/config.yaml --force
        """
    )
    parser.add_argument("--mode", 
                       choices=["surrogate-only", "dad_with_surrogate"],
                       default="dad_with_surrogate",
                       help="Training mode")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file")
    parser.add_argument("--force", action="store_true", help="Force retrain")
    parser.add_argument("--models-dir", default="models", help="Models directory")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    
    args = parser.parse_args()
    
    # Setup device
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
        print(f"ERROR: Config file '{args.config}' not found!")
        return 1
    
    cfg = load_config(args.config)
    
    print("\n" + "="*80)
    print("TRAINING PIPELINE")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Config: {args.config}")
    print(f"Models dir: {args.models_dir}")
    print(f"Device: {device.upper()}")
    print(f"Force retrain: {args.force}")
    print("="*80)
    
    try:
        # Stage 1: Train surrogate (always required)
        surrogate = train_surrogate_mode(cfg, device, args.models_dir, args.force)
        
        if args.mode == "dad_with_surrogate":
            # Stage 2: Generate fixed design
            fixed_design = train_fixed_design(cfg, surrogate, args.models_dir, args.force)
            
            # Stage 3: Train DAD policy
            if cfg.get("enable_dad", True):
                dad_policy = train_dad_policy_mode(cfg, surrogate, args.models_dir, args.force)
            else:
                print("\nWARNING: DAD disabled in config")
        
        # Summary
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"\nTrained models in: {args.models_dir}/")
        print("\nAvailable models:")
        for filename in sorted(os.listdir(args.models_dir)):
            filepath = os.path.join(args.models_dir, filename)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  {filename} ({size_mb:.2f} MB)")
        
        print("\nNext steps:")
        print(f"  python test.py --config {args.config} --episodes 100")
        
        return 0
        
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        exit(1)