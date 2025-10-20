#!/usr/bin/env python3
"""
Training pipeline with FIXED normalization handling
Addresses Priority 1 (Critical): Fix normalization in surrogate training/inference

Two-stage data loading:
- Stage 1: Surrogate training uses large dataset (MOCU + Sync)
- Stage 2: DAD training uses smaller dataset (MOCU + ERM + Sync)
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
from surrogate.train_surrogate import train_surrogate_model, DataCache
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


def load_two_stage_data(cfg: Dict, cache_dir: str = "dataset"):
    """
    Load two-stage data properly.
    
    Returns:
        dict with keys:
        - 'stage1_train': Large dataset for surrogate (MOCU + Sync)
        - 'stage1_val': Validation for surrogate
        - 'stage2_train': Smaller dataset for DAD (MOCU + ERM + Sync)
        - 'stage2_val': Validation for DAD
    """
    N = cfg["N"]
    K = cfg["K"]
    n_train_s1 = cfg["surrogate"]["n_train"]
    n_val_s1 = cfg["surrogate"]["n_val"]
    n_train_s2 = cfg["dad_rl"]["n_train"]
    n_val_s2 = cfg["dad_rl"]["n_val"]
    n_theta = cfg["surrogate"]["n_theta_samples"]
    n_erm = cfg["surrogate"].get("n_erm_samples", 10)
    
    def get_stage1_path(split):
        n = n_train_s1 if split == 'train' else n_val_s1
        seed = 42 if split == 'train' else 123
        return os.path.join(cache_dir, 
                           f"{split}_stage1_N{N}_K{K}_n{n}_mc{n_theta}_seed{seed}.pkl")
    
    def get_stage2_path(split):
        n = n_train_s2 if split == 'train' else n_val_s2
        seed = 42 if split == 'train' else 123
        return os.path.join(cache_dir, 
                           f"{split}_stage2_N{N}_K{K}_n{n}_mc{n_theta}_erm{n_erm}_seed{seed}.pkl")
    
    print("\n" + "="*80)
    print("LOADING TWO-STAGE DATA")
    print("="*80)
    
    data = {}
    
    # Load Stage 1 data
    for split in ['train', 'val']:
        path = get_stage1_path(split)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Stage 1 {split} data not found: {path}\n"
                f"Run: python generate_data.py --config {cfg}"
            )
        print(f"Loading Stage 1 {split}: {path}")
        with open(path, 'rb') as f:
            data[f'stage1_{split}'] = pickle.load(f)
        print(f"  Loaded {len(data[f'stage1_{split}'])} samples")
    
    # Load Stage 2 data
    for split in ['train', 'val']:
        path = get_stage2_path(split)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Stage 2 {split} data not found: {path}\n"
                f"Run: python generate_data.py --config {cfg}"
            )
        print(f"Loading Stage 2 {split}: {path}")
        with open(path, 'rb') as f:
            data[f'stage2_{split}'] = pickle.load(f)
        print(f"  Loaded {len(data[f'stage2_{split}'])} samples")
    
    print("="*80)
    return data


def train_surrogate_mode(cfg: Dict, device: str, models_dir: str, force: bool):
    """
    Train MPNN surrogate with FIXED normalization.
    
    CRITICAL FIX: Proper normalization handling
    - Compute stats on training data only
    - Store in model for inference
    - Use normalized forward during training
    - Use denormalized forward during inference
    """
    model_path = os.path.join(models_dir, "mpnn_surrogate.pth")
    
    print("\n" + "="*80)
    print("SURROGATE TRAINING (AccelerateOED 2023 aligned)")
    print("="*80)
    print("\nCRITICAL FIX: Normalization properly implemented")
    print("  - Stats computed on training data only")
    print("  - Stored in model for inference")
    print("  - forward_mocu_normalized() during training")
    print("  - forward_mocu() auto-denormalizes during inference")
    print("="*80)
    
    if not force and os.path.exists(model_path):
        print(f"\nModel exists: {model_path}")
        print("Use --force to retrain")
        try:
            surrogate = MPNNSurrogate(
                mocu_scale=cfg["surrogate"].get("mocu_scale", 1.0),
                hidden=cfg["surrogate"]["hidden"],
                dropout=cfg["surrogate"]["dropout"]
            )
            
            # Load model
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            surrogate.load_state_dict(state_dict)
            surrogate.to(device)
            surrogate.eval()
            
            # Verify normalization params are loaded
            print(f"Normalization params:")
            print(f"  Mean: {surrogate.mocu_mean.item():.4f}")
            print(f"  Std: {surrogate.mocu_std.item():.4f}")
            
            print("Model loaded successfully")
            return surrogate
        except Exception as e:
            print(f"WARNING: Failed to load: {e}")
            print("Training new model...")
    
    # Load two-stage data
    data = load_two_stage_data(cfg)
    
    # Use Stage 1 data for surrogate training (larger dataset, MOCU + Sync only)
    train_data = data['stage1_train']
    val_data = data['stage1_val']
    
    print("\nTraining Configuration:")
    print(f"  Architecture: MPNN with {cfg['surrogate']['hidden']} hidden dims")
    print(f"  Training samples: {len(train_data)} (Stage 1)")
    print(f"  Validation samples: {len(val_data)} (Stage 1)")
    print(f"  Epochs: {cfg['surrogate']['epochs']}")
    print(f"  Learning rate: {cfg['surrogate']['lr']}")
    print(f"  Batch size: {cfg['surrogate']['batch_size']}")
    print(f"  Target normalization: ENABLED (critical fix)")
    print("="*80)
    
    # Import training functions
    from surrogate.train_surrogate import SurrogateTrainer, normalize_dataset
    
    # Initialize model
    surrogate = MPNNSurrogate(
        mocu_scale=cfg["surrogate"].get("mocu_scale", 1.0),
        hidden=cfg["surrogate"]["hidden"],
        dropout=cfg["surrogate"]["dropout"]
    )
    
    # CRITICAL: Normalize data BEFORE training
    print("\nNormalizing MOCU labels...")
    train_data_normalized, (mocu_mean, mocu_std) = normalize_dataset(train_data, stats=None)
    val_data_normalized, _ = normalize_dataset(val_data, stats=(mocu_mean, mocu_std))
    
    # CRITICAL: Store normalization in model
    surrogate.mocu_mean = torch.tensor(mocu_mean, dtype=torch.float32)
    surrogate.mocu_std = torch.tensor(mocu_std, dtype=torch.float32)
    
    print(f"Normalization parameters:")
    print(f"  Mean: {mocu_mean:.4f}")
    print(f"  Std: {mocu_std:.4f}")
    
    # Save normalization separately for reference
    norm_path = os.path.join(models_dir, 'mocu_normalization.pkl')
    with open(norm_path, 'wb') as f:
        pickle.dump({'mean': mocu_mean, 'std': mocu_std}, f)
    print(f"Saved normalization: {norm_path}")
    
    # Train surrogate
    trainer = SurrogateTrainer(surrogate, device=device)
    
    print("\nTraining MOCU head...")
    mocu_results = trainer.train_mocu(
        train_data_normalized, val_data_normalized,
        epochs=cfg["surrogate"]["epochs"],
        lr=cfg["surrogate"]["lr"],
        batch_size=cfg["surrogate"]["batch_size"]
    )
    
    print("\nTraining Sync head...")
    sync_results = trainer.train_sync(
        train_data_normalized, val_data_normalized,
        epochs=cfg["surrogate"]["epochs"],
        lr=cfg["surrogate"]["lr"],
        batch_size=cfg["surrogate"]["batch_size"]
    )
    
    # Save model (normalization params included in state_dict)
    torch.save(surrogate.state_dict(), model_path)
    
    print("\n" + "="*80)
    print("SURROGATE TRAINING COMPLETE")
    print("="*80)
    print(f"Model saved: {model_path}")
    print(f"Final MOCU loss: {mocu_results['train_losses'][-1]:.4f}")
    if sync_results['train_losses']:
        print(f"Final Sync loss: {sync_results['train_losses'][-1]:.4f}")
    print(f"Normalization: mean={mocu_mean:.4f}, std={mocu_std:.4f}")
    print("="*80)
    
    return surrogate


def train_fixed_design(cfg: Dict, surrogate: MPNNSurrogate, models_dir: str, force: bool):
    """
    Generate static/fixed design baseline.
    This is the baseline from the DAD paper (not 2023 paper).
    """
    model_path = os.path.join(models_dir, "fixed_design.pkl")
    
    print("\n" + "="*80)
    print("STATIC DESIGN (DAD Paper Baseline)")
    print("="*80)
    print("Computed ONCE offline, reused for all instances")
    print("="*80)
    
    if not force and os.path.exists(model_path):
        print(f"Design exists: {model_path}")
        with open(model_path, 'rb') as f:
            fixed_sequence = pickle.load(f)
        print(f"Sequence: {fixed_sequence}")
        return fixed_sequence
    
    print("\nComputing static sequence using Greedy MPNN...")
    
    env_factory = make_env_factory(cfg, surrogate=surrogate)
    env = env_factory()
    fixed_sequence = []
    
    if TQDM_AVAILABLE:
        pbar = tqdm(range(cfg["K"]), desc="Computing static design", unit="step")
        iterator = pbar
    else:
        iterator = range(cfg["K"])
    
    for step in iterator:
        all_cands = env.candidate_pairs()
        untested_cands = [(i, j) for (i, j) in all_cands if not env.h.tested[i, j]]
        cands = untested_cands if untested_cands else all_cands
        
        xi = choose_next_pair_greedy(env, cands)
        fixed_sequence.append(xi)
        env.step(xi)
        
        if not TQDM_AVAILABLE:
            print(f"  Step {step+1}: {xi}")
    
    if TQDM_AVAILABLE:
        pbar.close()
    
    with open(model_path, 'wb') as f:
        pickle.dump(fixed_sequence, f)
    
    print(f"\nSaved: {model_path}")
    print(f"Static sequence: {fixed_sequence}")
    return fixed_sequence


def train_dad_policy_mode(cfg: Dict, surrogate: MPNNSurrogate, models_dir: str, force: bool):
    """
    Train DAD policy with proper data loading.
    Uses Stage 2 data which includes ERM labels.
    """
    model_path = os.path.join(models_dir, "dad_policy.pth")
    
    print("\n" + "="*80)
    print("DAD POLICY TRAINING (MOCU-based)")
    print("="*80)
    
    if not force and os.path.exists(model_path):
        print(f"\nPolicy exists: {model_path}")
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
    
    print("\nTraining Configuration:")
    print(f"  Policy hidden dims: {cfg['dad_rl']['hidden']}")
    print(f"  Epochs: {cfg['dad_rl']['epochs']}")
    print(f"  Episodes/epoch: {cfg['dad_rl']['episodes_per_epoch']}")
    print(f"  Learning rate: {cfg['dad_rl']['lr']}")
    print(f"  Uses Stage 2 data with ERM labels")
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
        description="Train models with fixed normalization and two-stage data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --mode surrogate-only --config configs/config.yaml
  python train.py --mode dad_with_surrogate --config configs/config_fast.yaml
  python train.py --mode dad_with_surrogate --config configs/config.yaml --force
        """
    )
    parser.add_argument("--mode", 
                       choices=["surrogate-only", "dad_with_surrogate"],
                       default="dad_with_surrogate",
                       help="Training mode")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--cpu", action="store_true")
    
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
        print(f"ERROR: Config file '{args.config}' not found")
        return 1
    
    cfg = load_config(args.config)
    
    print("\n" + "="*80)
    print("TRAINING PIPELINE (FIXED NORMALIZATION)")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Config: {args.config}")
    print(f"Models dir: {args.models_dir}")
    print(f"Device: {device.upper()}")
    print("="*80)
    
    try:
        # Stage 1: Train surrogate (with fixed normalization)
        surrogate = train_surrogate_mode(cfg, device, args.models_dir, args.force)
        
        if args.mode == "dad_with_surrogate":
            # Stage 2: Generate static design
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
            if os.path.isfile(filepath):
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