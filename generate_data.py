#!/usr/bin/env python3
"""
Standalone data generation script with caching and parallel processing.

Usage:
  # Generate training data
  python generate_data.py --config configs/config.yaml --split train
  
  # Generate validation data
  python generate_data.py --config configs/config.yaml --split val
  
  # Generate both
  python generate_data.py --config configs/config.yaml --split both
  
  # Use parallel processing (recommended)
  python generate_data.py --config configs/config.yaml --split both --parallel --workers 20
  
  # Force regenerate (ignore cache)
  python generate_data.py --config configs/config.yaml --split both --force
"""

import yaml
import numpy as np
import argparse
import os
import pickle
import time
from typing import Dict, List
from multiprocessing import Pool, cpu_count

from data_generation.synthetic_data import SyntheticDataGenerator

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: Install tqdm for progress bars: pip install tqdm")


class DataCache:
    """Manage cached datasets."""
    
    def __init__(self, cache_dir: str = "data"):
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
        print(f"✓ Saved {len(dataset)} samples to {path}")
        
        # Print file size
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  File size: {size_mb:.2f} MB")
    
    def load(self, split: str, N: int, K: int, n_samples: int, 
            n_theta_samples: int, seed: int) -> List[Dict]:
        """Load dataset from cache."""
        path = self.get_cache_path(split, N, K, n_samples, n_theta_samples, seed)
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
        print(f"✓ Loaded {len(dataset)} samples from {path}")
        
        # Print file size
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  File size: {size_mb:.2f} MB")
        return dataset


def generate_single_sample(args):
    """Generate a single sample (for parallel processing)."""
    seed, N, K, prior_bounds, omega_range, n_theta_samples, sim_opts = args
    
    try:
        rng = np.random.default_rng(seed)
        
        # Create a temporary generator for this sample
        generator = SyntheticDataGenerator(
            N=N, K=K, prior_bounds=prior_bounds, omega_range=omega_range,
            n_samples=1, sim_opts=sim_opts, n_theta_samples=n_theta_samples
        )
        
        omega = generator.generate_omega(rng)
        A_true = generator.generate_true_A(rng)
        data = generator.run_experiment_sequence(omega, A_true, rng)
        
        return data
    except Exception as e:
        return None


def generate_dataset_parallel(N: int, K: int, n_samples: int, n_theta_samples: int,
                              prior_bounds, omega_range, sim_opts, seed: int,
                              n_workers: int = None) -> List[Dict]:
    """Generate dataset using parallel processing with progress bar."""
    if n_workers is None:
        n_workers = min(cpu_count() - 2, 20)  # Leave 2 cores free
    
    print(f"Parallel data generation:")
    print(f"  Samples: {n_samples}")
    print(f"  MC samples per sample: {n_theta_samples}")
    print(f"  Workers: {n_workers}")
    print(f"  Total cores: {cpu_count()}")
    
    # Generate seeds for each sample
    rng = np.random.default_rng(seed)
    worker_seeds = rng.integers(0, 2**31, size=n_samples)
    
    # Prepare arguments for each worker
    worker_args = [
        (s, N, K, prior_bounds, omega_range, n_theta_samples, sim_opts)
        for s in worker_seeds
    ]
    
    # Generate samples in parallel with progress bar
    start_time = time.time()
    dataset = []
    
    with Pool(n_workers) as pool:
        if TQDM_AVAILABLE:
            # Use tqdm progress bar
            for result in tqdm(pool.imap_unordered(generate_single_sample, worker_args),
                             total=n_samples,
                             desc="Generating samples",
                             unit="sample",
                             ncols=100):
                if result is not None:
                    dataset.append(result)
        else:
            # Fallback to manual progress updates
            for i, result in enumerate(pool.imap_unordered(generate_single_sample, worker_args)):
                if result is not None:
                    dataset.append(result)
                
                # Progress update every 50 samples
                if (i + 1) % 50 == 0 or (i + 1) == n_samples:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    remaining = (n_samples - i - 1) / rate if rate > 0 else 0
                    progress = (i + 1) / n_samples * 100
                    print(f"  Progress: {i+1}/{n_samples} ({progress:.1f}%) - "
                          f"Rate: {rate:.2f} samples/sec - "
                          f"ETA: {remaining/60:.1f} min")
    
    elapsed = time.time() - start_time
    print(f"✓ Generated {len(dataset)} valid samples in {elapsed/60:.1f} minutes")
    print(f"  Average: {elapsed/len(dataset):.2f} sec/sample")
    
    return dataset


def generate_dataset_sequential(N: int, K: int, n_samples: int, n_theta_samples: int,
                                prior_bounds, omega_range, sim_opts, seed: int) -> List[Dict]:
    """Generate dataset sequentially with progress bar."""
    print(f"Sequential data generation:")
    print(f"  Samples: {n_samples}")
    print(f"  MC samples per sample: {n_theta_samples}")
    
    generator = SyntheticDataGenerator(
        N=N, K=K, prior_bounds=prior_bounds, omega_range=omega_range,
        n_samples=n_samples, sim_opts=sim_opts, n_theta_samples=n_theta_samples
    )
    
    # Override the generate_dataset method to add progress bar
    rng = np.random.default_rng(seed)
    dataset = []
    
    start_time = time.time()
    
    if TQDM_AVAILABLE:
        # Use tqdm progress bar
        iterator = tqdm(range(n_samples), 
                       desc="Generating samples",
                       unit="sample",
                       ncols=100)
    else:
        iterator = range(n_samples)
    
    for i in iterator:
        try:
            omega = generator.generate_omega(rng)
            A_true = generator.generate_true_A(rng)
            data = generator.run_experiment_sequence(omega, A_true, rng)
            dataset.append(data)
        except Exception as e:
            if not TQDM_AVAILABLE:
                print(f"Warning: Failed to generate sample {i}: {e}")
            continue
        
        # Manual progress update if no tqdm
        if not TQDM_AVAILABLE and ((i + 1) % 50 == 0 or (i + 1) == n_samples):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (n_samples - i - 1) / rate if rate > 0 else 0
            progress = (i + 1) / n_samples * 100
            print(f"  Progress: {i+1}/{n_samples} ({progress:.1f}%) - "
                  f"Rate: {rate:.2f} samples/sec - "
                  f"ETA: {remaining/60:.1f} min")
    
    elapsed = time.time() - start_time
    print(f"✓ Generated {len(dataset)} samples in {elapsed/60:.1f} minutes")
    print(f"  Average: {elapsed/len(dataset):.2f} sec/sample")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate training/validation data for Kuramoto experiment design",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate training data (parallel, recommended)
  python generate_data.py --split train --parallel
  
  # Generate both splits (parallel)
  python generate_data.py --split both --parallel --workers 20
  
  # Generate validation data only
  python generate_data.py --split val
  
  # Force regenerate (ignore cache)
  python generate_data.py --split both --force
  
  # Use custom config
  python generate_data.py --config configs/config_fast.yaml --split both --parallel
        """
    )
    parser.add_argument("--config", default="configs/config.yaml",
                       help="Config file")
    parser.add_argument("--split", choices=["train", "val", "both"], default="both",
                       help="Which split to generate")
    parser.add_argument("--parallel", action="store_true",
                       help="Use parallel processing (recommended)")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers (default: auto)")
    parser.add_argument("--force", action="store_true",
                       help="Force regenerate (ignore cache)")
    parser.add_argument("--cache-dir", default="data_cache",
                       help="Cache directory")
    
    args = parser.parse_args()
    
    # Load config
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found!")
        return 1
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Setup cache
    cache = DataCache(args.cache_dir)
    
    # Extract parameters
    N = cfg["N"]
    K = cfg["K"]
    prior_bounds = (cfg["prior_lower"], cfg["prior_upper"])
    omega_range = (cfg["omega"]["low"], cfg["omega"]["high"])
    sim_opts = cfg["sim"]
    
    n_train = cfg["surrogate"]["n_train"]
    n_val = cfg["surrogate"]["n_val"]
    n_theta_samples = cfg["surrogate"]["n_theta_samples"]
    
    print("="*80)
    print("DATA GENERATION")
    print("="*80)
    print(f"Configuration: N={N}, K={K}")
    print(f"Training samples: {n_train}")
    print(f"Validation samples: {n_val}")
    print(f"MC samples: {n_theta_samples}")
    print(f"Parallel: {args.parallel}")
    print(f"Cache dir: {args.cache_dir}")
    print("="*80)
    
    # Generate training data
    if args.split in ["train", "both"]:
        print("\n" + "="*80)
        print("TRAINING DATA")
        print("="*80)
        
        if not args.force and cache.exists("train", N, K, n_train, n_theta_samples, 42):
            print("⊙ Found cached training data")
            train_data = cache.load("train", N, K, n_train, n_theta_samples, 42)
        else:
            if args.parallel:
                train_data = generate_dataset_parallel(
                    N, K, n_train, n_theta_samples, prior_bounds, omega_range,
                    sim_opts, seed=42, n_workers=args.workers
                )
            else:
                train_data = generate_dataset_sequential(
                    N, K, n_train, n_theta_samples, prior_bounds, omega_range,
                    sim_opts, seed=42
                )
            
            cache.save(train_data, "train", N, K, n_train, n_theta_samples, 42)
    
    # Generate validation data
    if args.split in ["val", "both"]:
        print("\n" + "="*80)
        print("VALIDATION DATA")
        print("="*80)
        
        if not args.force and cache.exists("val", N, K, n_val, n_theta_samples, 123):
            print("⊙ Found cached validation data")
            val_data = cache.load("val", N, K, n_val, n_theta_samples, 123)
        else:
            if args.parallel:
                val_data = generate_dataset_parallel(
                    N, K, n_val, n_theta_samples, prior_bounds, omega_range,
                    sim_opts, seed=123, n_workers=args.workers
                )
            else:
                val_data = generate_dataset_sequential(
                    N, K, n_val, n_theta_samples, prior_bounds, omega_range,
                    sim_opts, seed=123
                )
            
            cache.save(val_data, "val", N, K, n_val, n_theta_samples, 123)
    
    print("\n" + "="*80)
    print("✓ DATA GENERATION COMPLETE")
    print("="*80)
    print(f"\nCached data saved in: {args.cache_dir}/")
    print("Use this data by running main.py with the same config")
    
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