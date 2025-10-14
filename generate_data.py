#!/usr/bin/env python3
"""
Fixed data generation script with better error handling and debugging.

Usage:
  # Generate training data
  python generate_data.py --config configs/config.yaml --split train
  
  # Generate both with parallel processing
  python generate_data.py --config configs/config.yaml --split both --parallel --workers 20
  
  # Debug mode (verbose output)
  python generate_data.py --config configs/config.yaml --split train --debug
"""

import yaml
import numpy as np
import argparse
import os
import pickle
import time
import sys
from typing import Dict, List, Tuple
from multiprocessing import Pool, cpu_count

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.belief import init_history, update_intervals, pair_threshold, build_belief_graph, History
from core.pacemaker_control import sync_check
from core.bisection import find_min_a_ctrl

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
        print(f" Saved {len(dataset)} samples to {path}")
        
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  File size: {size_mb:.2f} MB")
    
    def load(self, split: str, N: int, K: int, n_samples: int, 
            n_theta_samples: int, seed: int) -> List[Dict]:
        """Load dataset from cache."""
        path = self.get_cache_path(split, N, K, n_samples, n_theta_samples, seed)
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
        print(f" Loaded {len(dataset)} samples from {path}")
        
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  File size: {size_mb:.2f} MB")
        return dataset


class SimplifiedDataGenerator:
    """Simplified data generator that avoids complex nested functions."""
    
    def __init__(self, N: int, K: int, prior_bounds: Tuple[float, float],
                 omega_range: Tuple[float, float], sim_opts: Dict,
                 n_theta_samples: int = 20, debug: bool = False):
        self.N = N
        self.K = K
        self.prior_bounds = prior_bounds
        self.omega_range = omega_range
        self.sim_opts = sim_opts
        self.n_theta_samples = n_theta_samples
        self.debug = debug
    
    def generate_omega(self, rng: np.random.Generator) -> np.ndarray:
        """Generate natural frequencies."""
        return rng.uniform(self.omega_range[0], self.omega_range[1], size=self.N)
    
    def generate_true_A(self, rng: np.random.Generator) -> np.ndarray:
        """Generate true coupling matrix."""
        lo, hi = self.prior_bounds
        A = rng.uniform(lo, hi, size=(self.N, self.N))
        A = 0.5 * (A + A.T)
        np.fill_diagonal(A, 0.0)
        return A
    
    def sample_theta_from_belief(self, h: History, rng: np.random.Generator) -> np.ndarray:
        """Sample coupling matrix from belief."""
        A = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i+1, self.N):
                a_ij = rng.uniform(h.lower[i, j], h.upper[i, j])
                A[i, j] = a_ij
                A[j, i] = a_ij
        return A
    
    def find_optimal_control(self, A: np.ndarray, omega: np.ndarray) -> float:
        """Find optimal control parameter."""
        def check_fn(a_ctrl):
            try:
                return sync_check(A, omega, a_ctrl, **self.sim_opts)
            except:
                return False
        
        try:
            return find_min_a_ctrl(A, omega, check_fn, lo=0.0, hi_init=0.1, 
                                  tol=1e-3, verbose=False)
        except:
            return 2.0
    
    def compute_mocu(self, h: History, omega: np.ndarray, rng: np.random.Generator) -> float:
        """Compute MOCU via Monte Carlo estimation."""
        A_min = h.lower.copy()
        a_ibr = self.find_optimal_control(A_min, omega)
        
        mocu_sum = 0.0
        n_valid = 0
        
        for _ in range(self.n_theta_samples):
            A_sample = self.sample_theta_from_belief(h, rng)
            a_star = self.find_optimal_control(A_sample, omega)
            
            try:
                if sync_check(A_sample, omega, a_ibr, **self.sim_opts):
                    cost_ibr = a_ibr
                else:
                    cost_ibr = float('inf')
                
                cost_star = a_star
                
                if cost_ibr < float('inf'):
                    mocu_sum += (cost_ibr - cost_star)
                    n_valid += 1
            except:
                continue
        
        if n_valid > 0:
            return mocu_sum / n_valid
        else:
            return a_ibr * 0.5
    
    def compute_erm(self, h: History, xi: Tuple[int, int], omega: np.ndarray,
                   rng: np.random.Generator) -> float:
        """Compute ERM for candidate experiment."""
        i, j = xi
        lam = pair_threshold(omega, i, j)
        lo, up = h.lower[i, j], h.upper[i, j]
        
        lam_clamped = max(lo, min(lam, up))
        
        if up > lo:
            p_sync = max(0.0, (up - lam_clamped) / (up - lo))
        else:
            p_sync = 1.0 if lo >= lam else 0.0
        
        p_not_sync = 1.0 - p_sync
        
        erm = 0.0
        
        if p_sync > 1e-6:
            h_sync = self._copy_history(h)
            update_intervals(h_sync, xi, True, omega)
            mocu_sync = self.compute_mocu(h_sync, omega, rng)
            erm += p_sync * mocu_sync
        
        if p_not_sync > 1e-6:
            h_not = self._copy_history(h)
            update_intervals(h_not, xi, False, omega)
            mocu_not = self.compute_mocu(h_not, omega, rng)
            erm += p_not_sync * mocu_not
        
        return erm
    
    def _copy_history(self, h: History) -> History:
        """Deep copy history."""
        return History(
            pairs=h.pairs.copy(),
            outcomes=h.outcomes.copy(),
            lower=h.lower.copy(),
            upper=h.upper.copy(),
            tested=h.tested.copy()
        )
    
    def generate_single_sample(self, seed: int) -> Dict:
        """Generate a single training sample."""
        rng = np.random.default_rng(seed)
        
        try:
            omega = self.generate_omega(rng)
            A_true = self.generate_true_A(rng)
            h = init_history(self.N, self.prior_bounds)
            
            experiment_data = []
            
            # Run K experiments
            for step in range(self.K):
                
                belief_graph = build_belief_graph(h, omega)
                current_mocu = self.compute_mocu(h, omega, rng)
                
                candidates = [(i, j) for i in range(self.N) for j in range(i+1, self.N)
                             if not h.tested[i, j]]
                
                if not candidates:
                    candidates = [(i, j) for i in range(self.N) for j in range(i+1, self.N)]
                
                erm_scores = {}
                for cand_xi in candidates:
                    erm_scores[cand_xi] = self.compute_erm(h, cand_xi, omega, rng)
                
                xi = candidates[rng.integers(len(candidates))]
                i, j = xi
                lam = pair_threshold(omega, i, j)
                y_sync = (A_true[i, j] >= lam)
                
                step_data = {
                    'belief_graph': belief_graph,
                    'mocu': current_mocu,
                    'candidate_pairs': candidates,
                    'erm_scores': erm_scores,
                    'chosen_pair': xi,
                    'outcome': y_sync,
                    'step': step
                }
                experiment_data.append(step_data)
                
                update_intervals(h, xi, y_sync, omega)
            
            # Final state
            final_belief_graph = build_belief_graph(h, omega)
            final_mocu = self.compute_mocu(h, omega, rng)
            A_min = h.lower.copy()
            a_ctrl_star = self.find_optimal_control(A_min, omega)
            
            sync_scores = {}
            a_ctrl_values = np.linspace(0.0, 1.0, 10)
            for a_ctrl in a_ctrl_values:
                try:
                    sync_scores[float(a_ctrl)] = float(
                        sync_check(A_min, omega, a_ctrl, **self.sim_opts)
                    )
                except:
                    sync_scores[float(a_ctrl)] = 0.0
            
            return {
                'experiment_data': experiment_data,
                'final_belief_graph': final_belief_graph,
                'final_mocu': final_mocu,
                'a_ctrl_star': a_ctrl_star,
                'sync_scores': sync_scores,
                'omega': omega,
                'A_true': A_true
            }
            
        except Exception as e:
            if self.debug:
                print(f"Error generating sample {seed}: {e}")
                import traceback
                traceback.print_exc()
            return None


def worker_init(N, K, prior_bounds, omega_range, sim_opts, n_theta_samples, debug):
    """Initialize worker process with generator."""
    global _generator
    import warnings
    warnings.filterwarnings('ignore')
    
    # Ensure each worker uses CPU only (avoid CUDA issues)
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    _generator = SimplifiedDataGenerator(
        N=N, K=K, prior_bounds=prior_bounds, omega_range=omega_range,
        sim_opts=sim_opts, n_theta_samples=n_theta_samples, debug=debug
    )


def worker_process(seed):
    """Worker function for parallel processing."""
    try:
        global _generator
        return _generator.generate_single_sample(seed)
    except Exception as e:
        print(f"Worker error for seed {seed}: {e}")
        return None


def generate_dataset_parallel(N: int, K: int, n_samples: int, n_theta_samples: int,
                              prior_bounds, omega_range, sim_opts, seed: int,
                              n_workers: int = None, debug: bool = False) -> List[Dict]:
    """Generate dataset using parallel processing."""
    if n_workers is None:
        n_workers = min(cpu_count() - 2, 20)
    
    print(f"Parallel data generation:")
    print(f"  Samples: {n_samples}")
    print(f"  MC samples per sample: {n_theta_samples}")
    print(f"  Workers: {n_workers}")
    print(f"  Total cores: {cpu_count()}")
    print()
    
    # Test if a single sample works first
    print("Testing single sample generation...")
    test_gen = SimplifiedDataGenerator(
        N=N, K=K, prior_bounds=prior_bounds, omega_range=omega_range,
        sim_opts=sim_opts, n_theta_samples=n_theta_samples, debug=False
    )
    test_result = test_gen.generate_single_sample(seed)
    if test_result is None:
        print("ERROR: Test sample failed! Check errors above.")
        return []
    print("SUCCESS: Test sample succeeded!\n")
    
    rng = np.random.default_rng(seed)
    worker_seeds = rng.integers(0, 2**31, size=n_samples)
    
    start_time = time.time()
    dataset = []
    
    init_args = (N, K, prior_bounds, omega_range, sim_opts, n_theta_samples, debug)
    
    try:
        with Pool(n_workers, initializer=worker_init, initargs=init_args, maxtasksperchild=10) as pool:
            # Always use tqdm-style progress bar
            if TQDM_AVAILABLE:
                pbar = tqdm(total=n_samples, desc="Generating samples", 
                           unit="sample", ncols=100,
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
                
                for result in pool.imap(worker_process, worker_seeds, chunksize=5):
                    if result is not None:
                        dataset.append(result)
                    pbar.update(1)
                pbar.close()
            else:
                # Fallback: manual progress updates every 10 samples
                print(f"Generating {n_samples} samples...")
                for i, result in enumerate(pool.imap(worker_process, worker_seeds, chunksize=5)):
                    if result is not None:
                        dataset.append(result)
                    
                    if (i + 1) % 10 == 0 or (i + 1) == n_samples:
                        elapsed = time.time() - start_time
                        rate = (i + 1) / elapsed
                        remaining = (n_samples - i - 1) / rate if rate > 0 else 0
                        progress = (i + 1) / n_samples * 100
                        print(f"  [{progress:5.1f}%] {i+1}/{n_samples} samples | "
                              f"{rate:.2f} sample/s | ETA: {remaining/60:.1f} min")
    except KeyboardInterrupt:
        print("\n\n Interrupted by user")
        raise
    except Exception as e:
        print(f"\n Parallel processing failed: {e}")
        print("Falling back to sequential generation...")
        return generate_dataset_sequential(N, K, n_samples, n_theta_samples,
                                          prior_bounds, omega_range, sim_opts, seed, debug)
    
    elapsed = time.time() - start_time
    print(f"\n Generated {len(dataset)} valid samples in {elapsed/60:.1f} minutes")
    if len(dataset) > 0:
        print(f"  Average: {elapsed/len(dataset):.2f} sec/sample")
    
    return dataset


def generate_dataset_sequential(N: int, K: int, n_samples: int, n_theta_samples: int,
                                prior_bounds, omega_range, sim_opts, seed: int,
                                debug: bool = False) -> List[Dict]:
    """Generate dataset sequentially."""
    print(f"Sequential data generation:")
    print(f"  Samples: {n_samples}")
    print(f"  MC samples per sample: {n_theta_samples}")
    
    generator = SimplifiedDataGenerator(
        N=N, K=K, prior_bounds=prior_bounds, omega_range=omega_range,
        sim_opts=sim_opts, n_theta_samples=n_theta_samples, debug=debug
    )
    
    rng = np.random.default_rng(seed)
    worker_seeds = rng.integers(0, 2**31, size=n_samples)
    
    dataset = []
    start_time = time.time()
    
    if TQDM_AVAILABLE:
        pbar = tqdm(worker_seeds, desc="Generating samples", unit="sample", ncols=100,
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        for s in pbar:
            result = generator.generate_single_sample(s)
            if result is not None:
                dataset.append(result)
        pbar.close()
    else:
        print(f"Generating {n_samples} samples...")
        for i, s in enumerate(worker_seeds):
            result = generator.generate_single_sample(s)
            if result is not None:
                dataset.append(result)
            
            if (i + 1) % 10 == 0 or (i + 1) == n_samples:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (n_samples - i - 1) / rate if rate > 0 else 0
                progress = (i + 1) / n_samples * 100
                print(f"  [{progress:5.1f}%] {i+1}/{n_samples} samples | "
                      f"{rate:.2f} sample/s | ETA: {remaining/60:.1f} min")
    
    elapsed = time.time() - start_time
    print(f"\n Generated {len(dataset)} samples in {elapsed/60:.1f} minutes")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate training/validation data for Kuramoto experiment design",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Config file")
    parser.add_argument("--split", choices=["train", "val", "both"], default="both")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers")
    parser.add_argument("--force", action="store_true", help="Force regenerate")
    parser.add_argument("--cache-dir", default="dataset", help="Cache directory")
    parser.add_argument("--debug", action="store_true", help="Debug mode (forces sequential)")
    
    args = parser.parse_args()
    
    # Debug mode forces sequential processing
    if args.debug and args.parallel:
        print("� Warning: --debug flag forces sequential mode (disabling --parallel)")
        args.parallel = False
    
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
    print("DATA GENERATION")
    print("="*80)
    print(f"Configuration: N={N}, K={K}")
    print(f"Training samples: {n_train}, Validation samples: {n_val}")
    print(f"MC samples: {n_theta_samples}")
    print(f"Parallel: {args.parallel}, Debug: {args.debug}")
    print("="*80)
    
    # Generate training data
    if args.split in ["train", "both"]:
        print("\n" + "="*80)
        print("TRAINING DATA")
        print("="*80)
        
        if not args.force and cache.exists("train", N, K, n_train, n_theta_samples, 42):
            print(" Found cached training data")
            train_data = cache.load("train", N, K, n_train, n_theta_samples, 42)
        else:
            if args.parallel:
                train_data = generate_dataset_parallel(
                    N, K, n_train, n_theta_samples, prior_bounds, omega_range,
                    sim_opts, seed=42, n_workers=args.workers, debug=args.debug
                )
            else:
                train_data = generate_dataset_sequential(
                    N, K, n_train, n_theta_samples, prior_bounds, omega_range,
                    sim_opts, seed=42, debug=args.debug
                )
            
            cache.save(train_data, "train", N, K, n_train, n_theta_samples, 42)
    
    # Generate validation data
    if args.split in ["val", "both"]:
        print("\n" + "="*80)
        print("VALIDATION DATA")
        print("="*80)
        
        if not args.force and cache.exists("val", N, K, n_val, n_theta_samples, 123):
            print("� Found cached validation data")
            val_data = cache.load("val", N, K, n_val, n_theta_samples, 123)
        else:
            if args.parallel:
                val_data = generate_dataset_parallel(
                    N, K, n_val, n_theta_samples, prior_bounds, omega_range,
                    sim_opts, seed=123, n_workers=args.workers, debug=args.debug
                )
            else:
                val_data = generate_dataset_sequential(
                    N, K, n_val, n_theta_samples, prior_bounds, omega_range,
                    sim_opts, seed=123, debug=args.debug
                )
            
            cache.save(val_data, "val", N, K, n_val, n_theta_samples, 123)
    
    print("\n" + "="*80)
    print(" DATA GENERATION COMPLETE")
    print("="*80)
    print(f"\nCached data saved in: {args.cache_dir}/")
    
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