#!/usr/bin/env python3
"""
CORRECTED generate_data.py - Generate complete data (MOCU + ERM + Sync) for all methods.
"""

import yaml
import numpy as np
import argparse
import os
import pickle
import time
import sys
from typing import Dict, List, Tuple

# Ensure we can import from the project root
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_generation.synthetic_data import SyntheticDataGenerator

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class DataCache:
    """Simple cache manager for datasets."""
    
    def __init__(self, cache_dir: str = "dataset"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_path(self, split: str, N: int, K: int, n_samples: int, 
                       n_theta_samples: int, seed: int) -> str:
        """Generate cache filename."""
        filename = f"{split}_N{N}_K{K}_n{n_samples}_mc{n_theta_samples}_seed{seed}.pkl"
        return os.path.join(self.cache_dir, filename)
    
    def exists(self, split: str, N: int, K: int, n_samples: int, 
              n_theta_samples: int, seed: int) -> bool:
        """Check if cached data exists."""
        path = self.get_cache_path(split, N, K, n_samples, n_theta_samples, seed)
        return os.path.exists(path)
    
    def save(self, dataset: List[Dict], split: str, N: int, K: int, 
            n_samples: int, n_theta_samples: int, seed: int):
        """Save dataset to cache."""
        path = self.get_cache_path(split, N, K, n_samples, n_theta_samples, seed)
        try:
            with open(path, 'wb') as f:
                pickle.dump(dataset, f)
            print(f"SUCCESS: Saved {len(dataset)} samples to {path}")
            
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  File size: {size_mb:.2f} MB")
        except Exception as e:
            print(f"ERROR: Failed to save {path}: {e}")
    
    def load(self, split: str, N: int, K: int, n_samples: int, 
            n_theta_samples: int, seed: int) -> List[Dict]:
        """Load dataset from cache."""
        path = self.get_cache_path(split, N, K, n_samples, n_theta_samples, seed)
        try:
            with open(path, 'rb') as f:
                dataset = pickle.load(f)
            print(f"SUCCESS: Loaded {len(dataset)} samples from {path}")
            
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  File size: {size_mb:.2f} MB")
            return dataset
        except Exception as e:
            print(f"ERROR: Failed to load {path}: {e}")
            return []


def generate_complete_dataset(N: int, K: int, n_samples: int, n_theta_samples: int,
                             prior_bounds, omega_range, sim_opts, seed: int,
                             debug: bool = False) -> List[Dict]:
    """
    Generate complete dataset with MOCU + ERM + Sync labels.
    """
    print(f"\nGenerating complete dataset...")
    print(f"  Samples: {n_samples}")
    print(f"  MC samples per MOCU: {n_theta_samples}")
    print(f"  Computing: MOCU + ERM + Sync labels")
    print(f"  N={N}, K={K}, Seed={seed}")
    print()
    
    # Initialize generator with ERM computation enabled
    generator = SyntheticDataGenerator(
        N=N, K=K,
        prior_bounds=prior_bounds,
        omega_range=omega_range,
        n_samples=n_samples,
        sim_opts=sim_opts,
        n_theta_samples=n_theta_samples
    )
    
    # Enable all label computation
    generator.compute_true_erm = True
    generator.n_erm_samples = 10
    
    # Generate dataset
    try:
        dataset = generator.generate_dataset(seed=seed)
        return dataset
    except Exception as e:
        print(f"ERROR: Failed to generate dataset: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return []


def validate_dataset(dataset: List[Dict], min_samples: int = 10) -> bool:
    """Validate dataset quality."""
    if not dataset:
        print("\nERROR: Empty dataset!")
        return False
    
    if len(dataset) < min_samples:
        print(f"\nERROR: Only {len(dataset)} samples generated, need at least {min_samples}")
        return False
    
    print("\n" + "="*80)
    print("DATASET VALIDATION")
    print("="*80)
    
    try:
        # Check required keys
        required_keys = ['experiment_data', 'final_mocu', 'a_ctrl_star', 'omega', 'A_true']
        sample = dataset[0]
        missing_keys = [key for key in required_keys if key not in sample]
        
        if missing_keys:
            print(f"ERROR: Missing required keys: {missing_keys}")
            return False
        
        # Check MOCU statistics
        all_mocu = []
        all_erm = []
        for sample in dataset[:min(50, len(dataset))]:
            try:
                for step_data in sample['experiment_data']:
                    all_mocu.append(step_data['mocu'])
                    # Count ERM scores
                    all_erm.extend(list(step_data['erm_scores'].values()))
                all_mocu.append(sample['final_mocu'])
            except Exception as e:
                print(f"ERROR: Error extracting labels from sample: {e}")
                return False
        
        all_mocu = np.array(all_mocu)
        
        print(f"MOCU Statistics (first {min(50, len(dataset))} samples):")
        print(f"  Count: {len(all_mocu)}")
        print(f"  Mean: {all_mocu.mean():.4f}")
        print(f"  Std:  {all_mocu.std():.4f}")
        print(f"  Min:  {all_mocu.min():.4f}")
        print(f"  Max:  {all_mocu.max():.4f}")
        
        print(f"\nERM Statistics:")
        print(f"  Total ERM labels: {len(all_erm)}")
        if all_erm:
            all_erm = np.array(all_erm)
            print(f"  Mean: {all_erm.mean():.4f}")
            print(f"  Range: [{all_erm.min():.4f}, {all_erm.max():.4f}]")
        
        # Check for issues
        issues = []
        
        if all_mocu.std() < 0.001:
            issues.append("ERROR: MOCU values have very low variance")
        
        if np.any(all_mocu < 0):
            issues.append("ERROR: Some MOCU values are negative")
        
        if np.any(np.isnan(all_mocu)) or np.any(np.isinf(all_mocu)):
            issues.append("ERROR: Some MOCU values are NaN or infinite")
        
        if len(all_erm) == 0:
            issues.append("WARNING: No ERM labels found")
        
        # Check a_ctrl_star values
        a_ctrl_stars = [sample['a_ctrl_star'] for sample in dataset[:min(50, len(dataset))]]
        a_ctrl_stars = np.array(a_ctrl_stars)
        
        print(f"\na_ctrl_star Statistics:")
        print(f"  Mean: {a_ctrl_stars.mean():.4f}")
        print(f"  Range: [{a_ctrl_stars.min():.4f}, {a_ctrl_stars.max():.4f}]")
        
        if np.any(a_ctrl_stars <= 0):
            issues.append("ERROR: Some a_ctrl_star values are non-positive")
        
        if issues:
            print("\nIssues detected:")
            for issue in issues:
                print(f"  {issue}")
            return len([i for i in issues if i.startswith("ERROR")]) == 0
        else:
            print("\nSUCCESS: All validation checks passed!")
            return True
            
    except Exception as e:
        print(f"\nERROR: Validation failed with error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate complete training/validation data for Kuramoto experiment design",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_data.py --config configs/config.yaml --split both
  python generate_data.py --config configs/config_fast.yaml --split train --force
  python generate_data.py --split val --debug
        """
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--split", choices=["train", "val", "both"], default="both", 
                       help="Which data split to generate")
    parser.add_argument("--force", action="store_true", help="Force regenerate (overwrite cache)")
    parser.add_argument("--cache-dir", default="dataset", help="Cache directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Check config file
    if not os.path.exists(args.config):
        print(f"ERROR: Config file '{args.config}' not found!")
        print("\nAvailable configs:")
        for config_file in os.listdir("configs"):
            if config_file.endswith(".yaml"):
                print(f"  configs/{config_file}")
        return 1
    
    # Load config
    try:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        return 1
    
    cache = DataCache(args.cache_dir)
    
    # Extract config values
    N = cfg.get("N", 5)
    K = cfg.get("K", 4)
    prior_bounds = (cfg.get("prior_lower", 0.05), cfg.get("prior_upper", 0.50))
    omega_config = cfg.get("omega", {"low": -1.0, "high": 1.0})
    omega_range = (omega_config.get("low", -1.0), omega_config.get("high", 1.0))
    sim_opts = cfg.get("sim", {})
    surrogate_config = cfg.get("surrogate", {})
    n_train = surrogate_config.get("n_train", 1000)
    n_val = surrogate_config.get("n_val", 200)
    n_theta_samples = surrogate_config.get("n_theta_samples", 20)
    
    print("="*80)
    print("COMPLETE DATA GENERATION PIPELINE")
    print("="*80)
    print(f"Configuration: {args.config}")
    print(f"Split: {args.split}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Force regenerate: {args.force}")
    print(f"Debug mode: {args.debug}")
    print()
    print(f"Parameters: N={N}, K={K}")
    print(f"Training samples: {n_train}, Validation samples: {n_val}")
    print(f"MC samples: {n_theta_samples}")
    print("="*80)
    
    success = True
    
    # Generate training data
    if args.split in ["train", "both"]:
        print("\n" + "="*80)
        print("TRAINING DATA GENERATION")
        print("="*80)
        
        if not args.force and cache.exists("train", N, K, n_train, n_theta_samples, 42):
            print("SUCCESS: Found cached training data")
            train_data = cache.load("train", N, K, n_train, n_theta_samples, 42)
            if not train_data:
                print("ERROR: Failed to load cached training data")
                success = False
        else:
            print("Generating new training data...")
            train_data = generate_complete_dataset(
                N, K, n_train, n_theta_samples, prior_bounds, omega_range,
                sim_opts, seed=42, debug=args.debug
            )
            
            if train_data:
                if validate_dataset(train_data, min_samples=max(10, n_train // 100)):
                    cache.save(train_data, "train", N, K, n_train, n_theta_samples, 42)
                    print("SUCCESS: Training data generated successfully")
                else:
                    print("WARNING: Training data validation failed but continuing...")
                    cache.save(train_data, "train", N, K, n_train, n_theta_samples, 42)
            else:
                print("ERROR: Failed to generate training data")
                success = False
    
    # Generate validation data
    if args.split in ["val", "both"]:
        print("\n" + "="*80)
        print("VALIDATION DATA GENERATION")
        print("="*80)
        
        if not args.force and cache.exists("val", N, K, n_val, n_theta_samples, 123):
            print("SUCCESS: Found cached validation data")
            val_data = cache.load("val", N, K, n_val, n_theta_samples, 123)
            if not val_data:
                print("ERROR: Failed to load cached validation data")
                success = False
        else:
            print("Generating new validation data...")
            val_data = generate_complete_dataset(
                N, K, n_val, n_theta_samples, prior_bounds, omega_range,
                sim_opts, seed=123, debug=args.debug
            )
            
            if val_data:
                if validate_dataset(val_data, min_samples=max(10, n_val // 100)):
                    cache.save(val_data, "val", N, K, n_val, n_theta_samples, 123)
                    print("SUCCESS: Validation data generated successfully")
                else:
                    print("WARNING: Validation data validation failed but continuing...")
                    cache.save(val_data, "val", N, K, n_val, n_theta_samples, 123)
            else:
                print("ERROR: Failed to generate validation data")
                success = False
    
    # Final summary
    print("\n" + "="*80)
    if success:
        print("SUCCESS: DATA GENERATION COMPLETE")
        print("="*80)
        print(f"\nCached data location: {args.cache_dir}/")
        
        # List generated files
        if os.path.exists(args.cache_dir):
            files = [f for f in os.listdir(args.cache_dir) if f.endswith('.pkl')]
            if files:
                print("\nGenerated files:")
                for filename in sorted(files):
                    filepath = os.path.join(args.cache_dir, filename)
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    print(f"  {filename} ({size_mb:.2f} MB)")
        
        print("\nNext steps:")
        print(f"  python train.py --config {args.config} --methods all")
        return 0
    else:
        print("ERROR: DATA GENERATION FAILED")
        print("="*80)
        print("\nSome data generation steps failed. Check the errors above.")
        print("\nTroubleshooting:")
        print("  1. Check if all dependencies are installed")
        print("  2. Try running with --debug for more details")
        print("  3. Try the fast config: --config configs/config_fast.yaml")
        print("  4. Check if PyCUDA is properly installed for GPU acceleration")
        return 1


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\nWARNING: Interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\nERROR: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)