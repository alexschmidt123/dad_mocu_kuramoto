#!/usr/bin/env python3
"""
GPU-Accelerated Data Generation with PyCUDA for Kuramoto Experiment Design
Generates MOCU + ERM + Sync labels as required for DAD training
"""

import yaml
import numpy as np
import argparse
import os
import pickle
import time
import sys
from typing import Dict, List, Tuple

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.belief import init_history, update_intervals, pair_threshold, build_belief_graph, History
from core.pacemaker_control import find_optimal_control_batch, PYCUDA_AVAILABLE
from core.bisection import find_min_a_ctrl

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("WARNING: tqdm not available, progress bars disabled")

# Check PyCUDA availability
if not PYCUDA_AVAILABLE:
    print("="*80)
    print("WARNING: PyCUDA NOT AVAILABLE")
    print("="*80)
    print("Data generation will use CPU fallback (SLOW)")
    print("Install PyCUDA for GPU acceleration:")
    print("  conda install -c nvidia cuda-toolkit")
    print("  pip install pycuda")
    print("="*80)


class GPUBatchDataGenerator:
    """
    GPU-accelerated batch data generator using PyCUDA.
    Generates complete datasets with MOCU, ERM, and Sync labels.
    """
    
    def __init__(self, N: int, K: int, prior_bounds: Tuple[float, float],
                 omega_range: Tuple[float, float], sim_opts: Dict,
                 n_theta_samples: int = 20, n_erm_samples: int = 10):
        self.N = N
        self.K = K
        self.prior_bounds = prior_bounds
        self.omega_range = omega_range
        self.sim_opts = sim_opts
        self.n_theta_samples = n_theta_samples
        self.n_erm_samples = n_erm_samples
        self.use_gpu = PYCUDA_AVAILABLE
        
    def generate_omega_batch(self, batch_size: int, rng) -> np.ndarray:
        """Generate batch of natural frequencies"""
        return rng.uniform(self.omega_range[0], self.omega_range[1], 
                          size=(batch_size, self.N)).astype(np.float32)
    
    def generate_A_batch(self, batch_size: int, rng) -> np.ndarray:
        """Generate batch of coupling matrices"""
        lo, hi = self.prior_bounds
        A_batch = rng.uniform(lo, hi, size=(batch_size, self.N, self.N)).astype(np.float32)
        for i in range(batch_size):
            A_batch[i] = 0.5 * (A_batch[i] + A_batch[i].T)
            np.fill_diagonal(A_batch[i], 0.0)
        return A_batch
    
    def compute_mocu_batch_gpu(self, h_batch: List[History], omega_batch: np.ndarray, 
                               rng) -> np.ndarray:
        """
        Compute MOCU for a batch of belief states using GPU acceleration.
        MOCU = a*(A_min) - E[a*(theta)]
        """
        batch_size = len(h_batch)
        
        # Step 1: Compute worst-case control a*(A_min) for each belief state
        A_min_batch = np.array([h.lower for h in h_batch]).astype(np.float32)
        for i in range(batch_size):
            np.fill_diagonal(A_min_batch[i], 0.0)
        
        if self.use_gpu:
            a_worst_batch = find_optimal_control_batch(
                A_min_batch, omega_batch, self.sim_opts, use_gpu=True, tol=0.005
            )
        else:
            a_worst_batch = find_optimal_control_batch(
                A_min_batch, omega_batch, self.sim_opts, use_gpu=False, tol=0.005
            )
        
        # Step 2: Compute expected optimal control E[a*(theta)]
        mocu_values = []
        for i in range(batch_size):
            # Sample theta from belief
            theta_samples = []
            for _ in range(self.n_theta_samples):
                A_sample = self.sample_theta_from_belief(h_batch[i], rng)
                theta_samples.append(A_sample)
            
            theta_batch = np.array(theta_samples).astype(np.float32)
            omega_repeated = np.tile(omega_batch[i:i+1], (self.n_theta_samples, 1, 1))
            
            if self.use_gpu and self.n_theta_samples >= 10:
                a_optimal_batch = find_optimal_control_batch(
                    theta_batch, omega_repeated.squeeze(1), 
                    self.sim_opts, use_gpu=True, tol=0.005
                )
            else:
                a_optimal_batch = find_optimal_control_batch(
                    theta_batch, omega_repeated.squeeze(1),
                    self.sim_opts, use_gpu=False, tol=0.005
                )
            
            expected_optimal = np.mean(a_optimal_batch)
            mocu = max(0.0, min(5.0, a_worst_batch[i] - expected_optimal))
            mocu_values.append(mocu)
        
        return np.array(mocu_values)
    
    def sample_theta_from_belief(self, h: History, rng) -> np.ndarray:
        """Sample coupling matrix from current belief intervals"""
        A = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i+1, self.N):
                lower = max(h.lower[i, j], self.prior_bounds[0])
                upper = min(h.upper[i, j], self.prior_bounds[1])
                
                if upper <= lower:
                    a_ij = rng.uniform(self.prior_bounds[0], self.prior_bounds[1])
                else:
                    a_ij = rng.uniform(lower, upper)
                
                A[i, j] = a_ij
                A[j, i] = a_ij
        return A
    
    def compute_erm_for_candidate(self, h: History, xi: Tuple[int, int],
                                   omega: np.ndarray, rng) -> float:
        """Compute Expected Remaining MOCU for a candidate experiment"""
        import copy
        
        i, j = xi
        lam = pair_threshold(omega, i, j)
        
        posterior_mocu_values = []
        for _ in range(self.n_erm_samples):
            A_sample = self.sample_theta_from_belief(h, rng)
            y_sync = (A_sample[i, j] >= lam)
            
            h_posterior = copy.deepcopy(h)
            update_intervals(h_posterior, xi, y_sync, omega)
            
            # Compute MOCU for posterior (using smaller sample for speed)
            A_min_posterior = h_posterior.lower.copy()
            np.fill_diagonal(A_min_posterior, 0.0)
            
            from core.pacemaker_control import sync_check
            def check_fn(a_ctrl):
                try:
                    return sync_check(A_min_posterior, omega, a_ctrl, **self.sim_opts)
                except:
                    return False
            
            try:
                a_worst = find_min_a_ctrl(A_min_posterior, omega, check_fn, tol=0.005)
            except:
                a_worst = 2.0
            
            # Quick estimate of expected optimal
            theta_samples_erm = []
            for _ in range(min(5, self.n_theta_samples)):
                A_erm = self.sample_theta_from_belief(h_posterior, rng)
                
                def check_fn_erm(a_ctrl):
                    try:
                        return sync_check(A_erm, omega, a_ctrl, **self.sim_opts)
                    except:
                        return False
                
                try:
                    a_opt = find_min_a_ctrl(A_erm, omega, check_fn_erm, tol=0.005)
                    theta_samples_erm.append(a_opt)
                except:
                    theta_samples_erm.append(2.0)
            
            expected_opt = np.mean(theta_samples_erm) if theta_samples_erm else a_worst
            posterior_mocu = max(0.0, a_worst - expected_opt)
            posterior_mocu_values.append(posterior_mocu)
        
        return max(0.0, np.mean(posterior_mocu_values))
    
    def generate_sample(self, rng, sample_idx: int, total_samples: int) -> Dict:
        """Generate one complete sample with all labels"""
        omega = self.generate_omega_batch(1, rng)[0]
        A_true = self.generate_A_batch(1, rng)[0]
        
        h = init_history(self.N, self.prior_bounds)
        experiment_data = []
        
        for step in range(self.K):
            belief_graph = build_belief_graph(h, omega)
            
            # Compute current MOCU
            current_mocu_array = self.compute_mocu_batch_gpu([h], omega[np.newaxis, :], rng)
            current_mocu = float(current_mocu_array[0])
            
            # Get candidate pairs
            candidates = [(i, j) for i in range(self.N) for j in range(i+1, self.N)
                         if not h.tested[i, j]]
            
            if not candidates:
                candidates = [(i, j) for i in range(self.N) for j in range(i+1, self.N)]
            
            # Compute ERM for subset of candidates
            erm_scores = {}
            max_cands = min(10, len(candidates))
            selected_cands = rng.choice(len(candidates), size=max_cands, replace=False)
            
            for cand_idx in selected_cands:
                xi_cand = candidates[cand_idx]
                erm = self.compute_erm_for_candidate(h, xi_cand, omega, rng)
                erm_scores[xi_cand] = erm
            
            # Random selection for diversity
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
        final_mocu_array = self.compute_mocu_batch_gpu([h], omega[np.newaxis, :], rng)
        final_mocu = float(final_mocu_array[0])
        
        A_min = h.lower.copy()
        np.fill_diagonal(A_min, 0.0)
        
        from core.pacemaker_control import sync_check
        def check_fn(a_ctrl):
            try:
                return sync_check(A_min, omega, a_ctrl, **self.sim_opts)
            except:
                return False
        
        try:
            a_ctrl_star = find_min_a_ctrl(A_min, omega, check_fn, tol=0.005)
        except:
            a_ctrl_star = 2.0
        
        # Sync scores
        sync_scores = {}
        for a_ctrl in [0.05, 0.1, 0.15, 0.2, 0.3]:
            try:
                is_sync = sync_check(A_min, omega, a_ctrl, **self.sim_opts)
                sync_scores[a_ctrl] = 1 if is_sync else 0
            except:
                sync_scores[a_ctrl] = 0
        
        return {
            'experiment_data': experiment_data,
            'final_belief_graph': final_belief_graph,
            'final_mocu': final_mocu,
            'omega': omega,
            'A_true': A_true,
            'A_min': A_min,
            'a_ctrl_star': a_ctrl_star,
            'sync_scores': sync_scores
        }
    
    def generate_dataset(self, n_samples: int, seed: int) -> List[Dict]:
        """Generate complete dataset with progress bar"""
        rng = np.random.default_rng(seed)
        dataset = []
        
        print(f"\nGenerating {n_samples} samples...")
        print(f"  GPU Acceleration: {'ENABLED' if self.use_gpu else 'DISABLED'}")
        print(f"  MOCU MC samples: {self.n_theta_samples}")
        print(f"  ERM MC samples: {self.n_erm_samples}")
        
        if TQDM_AVAILABLE:
            pbar = tqdm(range(n_samples), desc="Generating data", unit="sample",
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            iterator = pbar
        else:
            iterator = range(n_samples)
            print()
        
        failed = 0
        for i in iterator:
            try:
                sample = self.generate_sample(rng, i, n_samples)
                dataset.append(sample)
            except Exception as e:
                failed += 1
                if not TQDM_AVAILABLE and (i + 1) % 50 == 0:
                    print(f"  Progress: {i+1}/{n_samples} (Failed: {failed})")
                continue
        
        if TQDM_AVAILABLE:
            pbar.close()
        
        print(f"\nGeneration complete: {len(dataset)}/{n_samples} successful (Failed: {failed})")
        return dataset


def validate_dataset(dataset: List[Dict], split_name: str) -> bool:
    """Validate dataset quality"""
    if not dataset:
        print(f"ERROR: Empty {split_name} dataset!")
        return False
    
    print(f"\n{'='*80}")
    print(f"{split_name.upper()} DATASET VALIDATION")
    print(f"{'='*80}")
    
    all_mocu = []
    all_erm = []
    
    for sample in dataset[:min(50, len(dataset))]:
        for step_data in sample['experiment_data']:
            all_mocu.append(step_data['mocu'])
            all_erm.extend(list(step_data['erm_scores'].values()))
        all_mocu.append(sample['final_mocu'])
    
    all_mocu = np.array(all_mocu)
    
    print(f"MOCU: count={len(all_mocu)}, mean={all_mocu.mean():.4f}, "
          f"range=[{all_mocu.min():.4f}, {all_mocu.max():.4f}]")
    
    if all_erm:
        all_erm = np.array(all_erm)
        print(f"ERM: count={len(all_erm)}, mean={all_erm.mean():.4f}, "
              f"range=[{all_erm.min():.4f}, {all_erm.max():.4f}]")
    
    issues = []
    if np.any(all_mocu < 0):
        issues.append("Negative MOCU values detected")
    if np.any(np.isnan(all_mocu)) or np.any(np.isinf(all_mocu)):
        issues.append("NaN or Inf MOCU values detected")
    
    if issues:
        for issue in issues:
            print(f"WARNING: {issue}")
        return False
    
    print("SUCCESS: Validation passed")
    return True


def main():
    parser = argparse.ArgumentParser(description="GPU-Accelerated Data Generation")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split", choices=["train", "val", "both"], default="both")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-dir", default="dataset")
    
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    N = cfg["N"]
    K = cfg["K"]
    prior_bounds = (cfg["prior_lower"], cfg["prior_upper"])
    omega_range = (cfg["omega"]["low"], cfg["omega"]["high"])
    sim_opts = cfg["sim"]
    n_train = cfg["surrogate"]["n_train"]
    n_val = cfg["surrogate"]["n_val"]
    n_theta_samples = cfg["surrogate"]["n_theta_samples"]
    
    generator = GPUBatchDataGenerator(
        N=N, K=K,
        prior_bounds=prior_bounds,
        omega_range=omega_range,
        sim_opts=sim_opts,
        n_theta_samples=n_theta_samples,
        n_erm_samples=cfg["surrogate"].get("n_erm_samples", 10)
    )
    
    print("="*80)
    print("GPU-ACCELERATED DATA GENERATION")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"PyCUDA: {'AVAILABLE' if PYCUDA_AVAILABLE else 'NOT AVAILABLE (using CPU)'}")
    print(f"N={N}, K={K}, train={n_train}, val={n_val}")
    print("="*80)
    
    def get_cache_path(split):
        return os.path.join(args.output_dir, 
                           f"{split}_N{N}_K{K}_n{n_train if split=='train' else n_val}_"
                           f"mc{n_theta_samples}_seed{42 if split=='train' else 123}.pkl")
    
    # Training data
    if args.split in ["train", "both"]:
        train_path = get_cache_path("train")
        if not args.force and os.path.exists(train_path):
            print(f"\nTraining data already exists: {train_path}")
        else:
            print("\nGenerating TRAINING data...")
            train_data = generator.generate_dataset(n_train, seed=42)
            if validate_dataset(train_data, "train"):
                with open(train_path, 'wb') as f:
                    pickle.dump(train_data, f)
                print(f"Saved: {train_path}")
    
    # Validation data
    if args.split in ["val", "both"]:
        val_path = get_cache_path("val")
        if not args.force and os.path.exists(val_path):
            print(f"\nValidation data already exists: {val_path}")
        else:
            print("\nGenerating VALIDATION data...")
            val_data = generator.generate_dataset(n_val, seed=123)
            if validate_dataset(val_data, "val"):
                with open(val_path, 'wb') as f:
                    pickle.dump(val_data, f)
                print(f"Saved: {val_path}")
    
    print("\n" + "="*80)
    print("DATA GENERATION COMPLETE")
    print("="*80)
    return 0


if __name__ == "__main__":
    exit(main())