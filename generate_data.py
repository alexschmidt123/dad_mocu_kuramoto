#!/usr/bin/env python3
"""
Sequential data generation following Chen et al. (2023) methodology.
Uses PyCUDA for GPU acceleration but NO multiprocessing.

This matches the AccelerateOED paper approach where data generation
is sequential but GPU-accelerated for MOCU computation.
"""

import yaml
import numpy as np
import argparse
import os
import pickle
import time
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generation.synthetic_data import SyntheticDataGenerator

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: Install tqdm for progress bars: pip install tqdm")


class DataCache:
    """Manage cached datasets."""
    
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
        with open(path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"SUCCESS: Saved {len(dataset)} samples to {path}")
        
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  File size: {size_mb:.2f} MB")
    
    def load(self, split: str, N: int, K: int, n_samples: int, 
            n_theta_samples: int, seed: int) -> List[Dict]:
        """Load dataset from cache."""
        path = self.get_cache_path(split, N, K, n_samples, n_theta_samples, seed)
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
        print(f"SUCCESS: Loaded {len(dataset)} samples from {path}")
        
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  File size: {size_mb:.2f} MB")
        return dataset


def generate_dataset_sequential(N: int, K: int, n_samples: int, n_theta_samples: int,
                                prior_bounds, omega_range, sim_opts, seed: int,
                                debug: bool = False) -> List[Dict]:
    """
    Sequential data generation following Chen et al. (2023).
    
    No multiprocessing - simple sequential loop with progress bar.
    GPU acceleration happens inside the generator for MOCU computation.
    """
    print("\n" + "="*80)
    print("SEQUENTIAL DATA GENERATION (Chen et al. 2023 approach)")
    print("="*80)
    print(f"Configuration:")
    print(f"  Samples: {n_samples}")
    print(f"  MC samples per MOCU: {n_theta_samples}")
    print(f"  N={N}, K={K}")
    print(f"  GPU acceleration: Enabled for MOCU computation")
    print("="*80)
    print()
    
    # Initialize generator
    generator = SyntheticDataGenerator(
        N=N, K=K,
        prior_bounds=prior_bounds,
        omega_range=omega_range,
        n_samples=n_samples,
        sim_opts=sim_opts,
        n_theta_samples=n_theta_samples
    )
    
    # Generate dataset sequentially
    dataset = []
    start_time = time.time()
    
    # Progress bar setup
    if TQDM_AVAILABLE:
        pbar = tqdm(range(n_samples), desc="Generating samples", 
                   unit="sample", ncols=100,
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        iterator = pbar
    else:
        iterator = range(n_samples)
        print(f"Generating {n_samples} samples...")
    
    # Sequential generation
    rng = np.random.default_rng(seed)
    
    for i in iterator:
        # Generate omega and true coupling matrix
        omega = generator.generate_omega(rng)
        A_true = generator.generate_true_A(rng)
        
        try:
            # Run complete experiment sequence
            data = generator.run_experiment_sequence(omega, A_true, rng)
            dataset.append(data)
            
        except Exception as e:
            if debug:
                print(f"\nWarning: Failed sample {i}: {e}")
                import traceback
                traceback.print_exc()
            continue
        
        # Progress update for non-tqdm
        if not TQDM_AVAILABLE and ((i + 1) % 50 == 0 or (i + 1) == n_samples):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (n_samples - i - 1) / rate if rate > 0 else 0
            progress = (i + 1) / n_samples * 100
            print(f"  [{progress:5.1f}%] {i+1}/{n_samples} samples | "
                  f"{rate:.2f} sample/s | ETA: {remaining/60:.1f} min")
    
    if TQDM_AVAILABLE:
        pbar.close()
    
    elapsed = time.time() - start_time
    print(f"\nSUCCESS: Generated {len(dataset)} valid samples in {elapsed/60:.1f} minutes")
    if len(dataset) > 0:
        print(f"  Average: {elapsed/len(dataset):.2f} sec/sample")
    
    return dataset


def validate_dataset(dataset: List[Dict]):
    """Validate dataset quality."""
    if not dataset:
        print("\nERROR: Empty dataset!")
        return False
    
    print("\n" + "="*80)
    print("DATASET VALIDATION")
    print("="*80)
    
    # Check MOCU statistics
    all_mocu = []
    for sample in dataset[:min(50, len(dataset))]:
        for step_data in sample['experiment_data']:
            all_mocu.append(step_data['mocu'])
        all_mocu.append(sample['final_mocu'])
    
    all_mocu = np.array(all_mocu)
    
    print(f"MOCU Statistics (first {min(50, len(dataset))} samples):")
    print(f"  Mean: {all_mocu.mean():.4f}")
    print(f"  Std:  {all_mocu.std():.4f}")
    print(f"  Min:  {all_mocu.min():.4f}")
    print(f"  Max:  {all_mocu.max():.4f}")
    print(f"  Median: {np.median(all_mocu):.4f}")
    
    # Check for issues
    issues = []
    
    if all_mocu.std() < 0.01:
        issues.append("FAIL: Labels have very low variance - model may not learn")
    
    if all_mocu.min() < 0:
        issues.append("FAIL: Some MOCU values are negative - computation error!")
    
    if all_mocu.max() > 10.0:
        issues.append("WARNING: Some MOCU values very large - may need scaling")
    
    if issues:
        print("\nIssues detected:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("\nSUCCESS: All validation checks passed!")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate training/validation data (Sequential, Chen et al. 2023 approach)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Config file")
    parser.add_argument("--split", choices=["train", "val", "both"], default="both")
    parser.add_argument("--force", action="store_true", help="Force regenerate")
    parser.add_argument("--cache-dir", default="dataset", help="Cache directory")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found!")
        return 1
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    cache = DataCache(args.cache_dir)
    
    N = cfg["N"]
    K = cfg["K"]
    prior_bounds = (cfg["prior_lower"], cfg["prior_upper"])
    omega_range = (cfg["omega"]["low"], cfg["omega"]["high"])
    sim_opts = cfg["sim"]
    n_train = cfg["surrogate"]["n_train"]
    n_val = cfg["surrogate"]["n_val"]
    n_theta_samples = cfg["surrogate"]["n_theta_samples"]
    
    print("="*80)
    print("SEQUENTIAL DATA GENERATION")
    print("="*80)
    print(f"Configuration: N={N}, K={K}")
    print(f"Training samples: {n_train}, Validation samples: {n_val}")
    print(f"MC samples: {n_theta_samples}")
    print(f"Method: Sequential (Chen et al. 2023)")
    print(f"Debug: {args.debug}")
    print("="*80)
    
    # Generate training data
    if args.split in ["train", "both"]:
        print("\n" + "="*80)
        print("TRAINING DATA")
        print("="*80)
        
        if not args.force and cache.exists("train", N, K, n_train, n_theta_samples, 42):
            print("SUCCESS: Found cached training data")
            train_data = cache.load("train", N, K, n_train, n_theta_samples, 42)
        else:
            train_data = generate_dataset_sequential(
                N, K, n_train, n_theta_samples, prior_bounds, omega_range,
                sim_opts, seed=42, debug=args.debug
            )
            
            if train_data:
                if validate_dataset(train_data):
                    cache.save(train_data, "train", N, K, n_train, n_theta_samples, 42)
                else:
                    print("\nWARNING: Dataset validation failed but saving anyway")
                    cache.save(train_data, "train", N, K, n_train, n_theta_samples, 42)
    
    # Generate validation data
    if args.split in ["val", "both"]:
        print("\n" + "="*80)
        print("VALIDATION DATA")
        print("="*80)
        
        if not args.force and cache.exists("val", N, K, n_val, n_theta_samples, 123):
            print("SUCCESS: Found cached validation data")
            val_data = cache.load("val", N, K, n_val, n_theta_samples, 123)
        else:
            val_data = generate_dataset_sequential(
                N, K, n_val, n_theta_samples, prior_bounds, omega_range,
                sim_opts, seed=123, debug=args.debug
            )
            
            if val_data:
                if validate_dataset(val_data):
                    cache.save(val_data, "val", N, K, n_val, n_theta_samples, 123)
                else:
                    print("\nWARNING: Dataset validation failed but saving anyway")
                    cache.save(val_data, "val", N, K, n_val, n_theta_samples, 123)
    
    print("\n" + "="*80)
    print("DATA GENERATION COMPLETE")
    print("="*80)
    print(f"\nCached data saved in: {args.cache_dir}/")
    print("\nNext steps:")
    print(f"  python train.py --config {args.config} --methods all")
    
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