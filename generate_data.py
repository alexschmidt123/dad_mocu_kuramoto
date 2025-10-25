#!/usr/bin/env python3
"""
Complete Two-Stage GPU-Accelerated Data Generation
Stage 1: Surrogate training data (MOCU + Sync only)
Stage 2: DAD training data (MOCU + ERM + Sync)

Usage: python generate_data.py --config configs/config_fast.yaml
Generates ALL data (train + val, Stage 1 + Stage 2) automatically
"""

import yaml
import numpy as np
import argparse
import os
import pickle
import time
import sys
from typing import Dict, List, Tuple
from pathlib import Path

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.belief import init_history, update_intervals, pair_threshold, build_belief_graph, History
from core.pacemaker_control import find_optimal_control_batch, PYCUDA_AVAILABLE, sync_check
from core.bisection import find_min_a_ctrl

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("WARNING: tqdm not available, progress bars disabled")

# Check PyCUDA
if not PYCUDA_AVAILABLE:
    print("="*80)
    print("WARNING: PyCUDA NOT AVAILABLE")
    print("="*80)
    print("Data generation will use CPU fallback (VERY SLOW)")
    print("Install PyCUDA for GPU acceleration:")
    print("  conda install -c nvidia cuda-toolkit")
    print("  pip install pycuda")
    print("="*80)
else:
    print("="*80)
    print("PyCUDA AVAILABLE - GPU acceleration ENABLED")
    print("="*80)


class TwoStageDataGenerator:
    """
    Two-Stage Data Generator following AccelerateOED 2023 approach.
    
    Stage 1: Large dataset for surrogate training
    - Generates: MOCU + Sync only
    - Purpose: Train accurate MOCU predictor
    - Size: Larger (e.g., 2000 train, 400 val)
    
    Stage 2: Smaller dataset for DAD training  
    - Generates: MOCU + ERM + Sync
    - Purpose: Train adaptive policy
    - Size: Smaller (e.g., 500 train, 100 val)
    - Uses Stage 1 data + adds ERM labels
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
        Compute MOCU for a batch using paper's sampling approach.
        
        For each belief state:
        1. Sample n_theta_samples coupling matrices
        2. Find optimal control for each sample
        3. Compute MOCU = max(a_optimal) - mean(a_optimal)
        """
        batch_size = len(h_batch)
        mocu_values = []
        
        for i in range(batch_size):
            h = h_batch[i]
            omega = omega_batch[i]
            
            # Sample coupling matrices from belief
            theta_samples = []
            for _ in range(self.n_theta_samples):
                A_sample = self.sample_theta_from_belief(h, rng)
                theta_samples.append(A_sample)
            
            # Compute optimal controls
            theta_batch = np.array(theta_samples).astype(np.float32)
            omega_repeated = np.tile(omega, (self.n_theta_samples, 1)).astype(np.float32)
            
            # Use GPU if available and batch is large enough
            if self.use_gpu and self.n_theta_samples >= 10:
                try:
                    a_optimal_batch = find_optimal_control_batch(
                        theta_batch, omega_repeated, 
                        self.sim_opts, use_gpu=True, tol=0.005
                    )
                except:
                    # Fallback to CPU
                    a_optimal_batch = find_optimal_control_batch(
                        theta_batch, omega_repeated,
                        self.sim_opts, use_gpu=False, tol=0.005
                    )
            else:
                a_optimal_batch = find_optimal_control_batch(
                    theta_batch, omega_repeated,
                    self.sim_opts, use_gpu=False, tol=0.005
                )
            
            # Compute MOCU using paper's formula
            K_max = self.n_theta_samples
            a_save = a_optimal_batch
            
            if K_max >= 1000:
                temp = np.sort(a_save)
                ll = int(K_max * 0.005)
                uu = int(K_max * 0.995)
                a_save_trimmed = temp[ll:uu]
                
                if len(a_save_trimmed) > 0:
                    a_star = np.max(a_save_trimmed)
                    mocu = np.sum(a_star - a_save_trimmed) / (K_max * 0.99)
                else:
                    mocu = 0.1
            else:
                a_star = np.max(a_save)
                mocu = np.sum(a_star - a_save) / K_max
            
            mocu = max(0.0, min(5.0, mocu))
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
            
            # Compute MOCU for posterior
            A_min_posterior = h_posterior.lower.copy()
            np.fill_diagonal(A_min_posterior, 0.0)
            
            def check_fn(a_ctrl):
                try:
                    return sync_check(A_min_posterior, omega, a_ctrl, **self.sim_opts)
                except:
                    return False
            
            try:
                a_worst = find_min_a_ctrl(A_min_posterior, omega, check_fn, 
                                         tol=0.005, verbose=False)
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
                    a_opt = find_min_a_ctrl(A_erm, omega, check_fn_erm, 
                                           tol=0.005, verbose=False)
                    theta_samples_erm.append(a_opt)
                except:
                    theta_samples_erm.append(2.0)
            
            expected_opt = np.mean(theta_samples_erm) if theta_samples_erm else a_worst
            posterior_mocu = max(0.0, a_worst - expected_opt)
            posterior_mocu_values.append(posterior_mocu)
        
        return max(0.0, np.mean(posterior_mocu_values))
    
    def generate_stage1_sample(self, rng) -> Dict:
        """Generate Stage 1 sample with adaptive bounds"""
        omega = self.generate_omega_batch(1, rng)[0]
        
        # CRITICAL: Use adaptive bounds instead of fixed prior_bounds
        aLower, aUpper = self.compute_adaptive_bounds(omega)
        
        # Sample true A within these adaptive bounds
        A_true = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i+1, self.N):
                A_true[i, j] = rng.uniform(aLower[i, j], aUpper[i, j])
                A_true[j, i] = A_true[i, j]
        
        # Initialize history with adaptive bounds
        h = History(
            pairs=[],
            outcomes=[],
            lower=aLower.copy(),
            upper=aUpper.copy(),
            tested=np.zeros((self.N, self.N), dtype=bool)
        )
        
        # Rest of the method continues as before...
        experiment_data = []
        
        for step in range(self.K):
            belief_graph = build_belief_graph(h, omega)
            
            # Compute current MOCU with corrected formula
            current_mocu_array = self.compute_mocu_batch_gpu([h], omega[np.newaxis, :], rng)
            current_mocu = float(current_mocu_array[0])
            # Get candidate pairs
            candidates = [(i, j) for i in range(self.N) for j in range(i+1, self.N)
                         if not h.tested[i, j]]
            
            if not candidates:
                candidates = [(i, j) for i in range(self.N) for j in range(i+1, self.N)]
            
            # Random selection for diversity
            xi = candidates[rng.integers(len(candidates))]
            i, j = xi
            lam = pair_threshold(omega, i, j)
            y_sync = (A_true[i, j] >= lam)
            
            step_data = {
                'belief_graph': belief_graph,
                'mocu': current_mocu,
                'candidate_pairs': candidates,
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
        
        def check_fn(a_ctrl):
            try:
                return sync_check(A_min, omega, a_ctrl, **self.sim_opts)
            except:
                return False
        
        try:
            a_ctrl_star = find_min_a_ctrl(A_min, omega, check_fn, 
                                          tol=0.005, verbose=False)
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
            'sync_scores': sync_scores,
            'stage': 1  # Mark as Stage 1 data
        }
    
    def compute_adaptive_bounds(self, omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute adaptive prior bounds exactly matching paper2023's runMainForPerformanceMeasure.py
        """
        N = len(omega)
        aUpper = np.zeros((N, N))
        aLower = np.zeros((N, N))
        
        # Step 1: Basic bounds based on synchronization threshold
        for i in range(N):
            for j in range(i + 1, N):
                syncThreshold = 0.5 * np.abs(omega[i] - omega[j])
                aUpper[i, j] = syncThreshold * 1.15
                aLower[i, j] = syncThreshold * 0.85
                aUpper[j, i] = aUpper[i, j]
                aLower[j, i] = aLower[i, j]
        
        # Step 2: Paper2023's heterogeneous structure for N=5
        if N == 5:
            # Weaker coupling for pairs (0,2), (0,3), (0,4)
            aUpper[0, 2:5] = aUpper[0, 2:5] * 0.3
            aLower[0, 2:5] = aLower[0, 2:5] * 0.3
            
            # Medium coupling for pairs (1,3), (1,4)  
            aUpper[1, 3:5] = aUpper[1, 3:5] * 0.45
            aLower[1, 3:5] = aLower[1, 3:5] * 0.45
            
            # Make symmetric
            for i in [0]:
                for j in range(2, 5):
                    aUpper[j, i] = aUpper[i, j]
                    aLower[j, i] = aLower[i, j]
            
            for i in [1]:
                for j in range(3, 5):
                    aUpper[j, i] = aUpper[i, j]
                    aLower[j, i] = aLower[i, j]
        
        return aLower, aUpper

    def add_erm_to_sample(self, sample: Dict, rng) -> Dict:
        """
        Add ERM labels to Stage 1 sample to create Stage 2 sample.
        This is more efficient than regenerating everything.
        """
        omega = sample['omega']
        A_true = sample['A_true']
        
        # Replay experiment sequence and add ERM
        h = init_history(self.N, self.prior_bounds)
        
        for step_idx, step_data in enumerate(sample['experiment_data']):
            # Compute ERM for candidates at this step
            candidates = step_data['candidate_pairs']
            erm_scores = {}
            
            max_cands = min(10, len(candidates))
            selected_cands = rng.choice(len(candidates), size=max_cands, replace=False)
            
            for cand_idx in selected_cands:
                xi_cand = candidates[cand_idx]
                erm = self.compute_erm_for_candidate(h, xi_cand, omega, rng)
                erm_scores[xi_cand] = erm
            
            # Add ERM to step data
            step_data['erm_scores'] = erm_scores
            
            # Update belief to next step
            xi = step_data['chosen_pair']
            y_sync = step_data['outcome']
            update_intervals(h, xi, y_sync, omega)
        
        sample['stage'] = 2  # Mark as Stage 2 data
        return sample
    
    def generate_stage1_dataset(self, n_samples: int, seed: int, 
                               desc: str = "Stage 1") -> List[Dict]:
        """Generate Stage 1 dataset (MOCU + Sync only)"""
        rng = np.random.default_rng(seed)
        dataset = []
        
        print(f"\n{desc} (MOCU + Sync):")
        print(f"  Samples: {n_samples}")
        print(f"  MOCU MC samples: {self.n_theta_samples}")
        print(f"  GPU: {'ENABLED' if self.use_gpu else 'DISABLED'}")
        
        if TQDM_AVAILABLE:
            pbar = tqdm(range(n_samples), desc=f"Generating {desc}", unit="sample",
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            iterator = pbar
        else:
            iterator = range(n_samples)
        
        failed = 0
        for i in iterator:
            try:
                sample = self.generate_stage1_sample(rng)
                dataset.append(sample)
            except Exception as e:
                failed += 1
                if not TQDM_AVAILABLE and (i + 1) % 50 == 0:
                    print(f"  Progress: {i+1}/{n_samples} (Failed: {failed})")
                continue
        
        if TQDM_AVAILABLE:
            pbar.close()
        
        print(f"  Complete: {len(dataset)}/{n_samples} successful (Failed: {failed})")
        return dataset
    
    def generate_stage2_dataset(self, stage1_data: List[Dict], n_samples: int, 
                               seed: int, desc: str = "Stage 2") -> List[Dict]:
        """
        Generate Stage 2 dataset by adding ERM to Stage 1 samples.
        Takes subset of Stage 1 and adds ERM labels.
        """
        rng = np.random.default_rng(seed)
        
        # Use subset of Stage 1 data
        n_use = min(n_samples, len(stage1_data))
        selected_indices = rng.choice(len(stage1_data), size=n_use, replace=False)
        
        print(f"\n{desc} (MOCU + ERM + Sync):")
        print(f"  Samples: {n_use}")
        print(f"  ERM MC samples: {self.n_erm_samples}")
        print(f"  Adding ERM to Stage 1 data...")
        
        dataset = []
        
        if TQDM_AVAILABLE:
            pbar = tqdm(selected_indices, desc=f"Adding ERM to {desc}", unit="sample",
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            iterator = pbar
        else:
            iterator = selected_indices
        
        for idx in iterator:
            try:
                # Deep copy to avoid modifying original
                import copy
                sample = copy.deepcopy(stage1_data[idx])
                sample_with_erm = self.add_erm_to_sample(sample, rng)
                dataset.append(sample_with_erm)
            except Exception as e:
                continue
        
        if TQDM_AVAILABLE:
            pbar.close()
        
        print(f"  Complete: {len(dataset)}/{n_use} successful")
        return dataset


def validate_dataset(dataset: List[Dict], split_name: str, stage: int) -> bool:
    """Validate dataset quality"""
    if not dataset:
        print(f"ERROR: Empty {split_name} dataset")
        return False
    
    print(f"\n{'='*80}")
    print(f"{split_name.upper()} STAGE {stage} VALIDATION")
    print(f"{'='*80}")
    
    all_mocu = []
    all_erm = []
    all_sync = []
    
    for sample in dataset[:min(50, len(dataset))]:
        for step_data in sample['experiment_data']:
            all_mocu.append(step_data['mocu'])
            if 'erm_scores' in step_data:
                all_erm.extend(list(step_data['erm_scores'].values()))
        all_mocu.append(sample['final_mocu'])
        if 'sync_scores' in sample:
            all_sync.extend(list(sample['sync_scores'].values()))
    
    all_mocu = np.array(all_mocu)
    
    print(f"MOCU: count={len(all_mocu)}, mean={all_mocu.mean():.4f}, "
          f"range=[{all_mocu.min():.4f}, {all_mocu.max():.4f}]")
    
    if all_erm:
        all_erm = np.array(all_erm)
        print(f"ERM: count={len(all_erm)}, mean={all_erm.mean():.4f}, "
              f"range=[{all_erm.min():.4f}, {all_erm.max():.4f}]")
    
    if all_sync:
        all_sync = np.array(all_sync)
        print(f"Sync: count={len(all_sync)}, mean={all_sync.mean():.2f}")
    
    issues = []
    if np.any(all_mocu < 0):
        issues.append("Negative MOCU values")
    if np.any(np.isnan(all_mocu)) or np.any(np.isinf(all_mocu)):
        issues.append("NaN or Inf MOCU values")
    
    if issues:
        for issue in issues:
            print(f"WARNING: {issue}")
        return False
    
    print("SUCCESS: Validation passed")
    return True



def main():
    parser = argparse.ArgumentParser(
        description="Two-Stage GPU-Accelerated Data Generation",
        epilog="Usage: python generate_data.py --config configs/config_fast.yaml"
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--force", action="store_true", 
                       help="Force regenerate (overwrite existing)")
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
    
    # Stage 1: Surrogate training data
    n_train_stage1 = cfg["surrogate"]["n_train"]
    n_val_stage1 = cfg["surrogate"]["n_val"]
    n_theta_samples = cfg["surrogate"]["n_theta_samples"]
    
    # Stage 2: DAD training data
    n_train_stage2 = cfg["dad_rl"]["n_train"]
    n_val_stage2 = cfg["dad_rl"]["n_val"]
    n_erm_samples = cfg["surrogate"].get("n_erm_samples", 10)
    
    generator = TwoStageDataGenerator(
        N=N, K=K,
        prior_bounds=prior_bounds,
        omega_range=omega_range,
        sim_opts=sim_opts,
        n_theta_samples=n_theta_samples,
        n_erm_samples=n_erm_samples
    )
    
    print("="*80)
    print("TWO-STAGE DATA GENERATION")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"PyCUDA: {'AVAILABLE' if PYCUDA_AVAILABLE else 'NOT AVAILABLE (CPU)'}")
    print(f"N={N}, K={K}")
    print(f"\nStage 1 (Surrogate): train={n_train_stage1}, val={n_val_stage1}")
    print(f"Stage 2 (DAD): train={n_train_stage2}, val={n_val_stage2}")
    print("="*80)
    
    def get_cache_path(split, stage):
        if stage == 1:
            n_samples = n_train_stage1 if split=='train' else n_val_stage1
            return os.path.join(args.output_dir, 
                               f"{split}_stage1_N{N}_K{K}_n{n_samples}_"
                               f"mc{n_theta_samples}_seed{42 if split=='train' else 123}.pkl")
        else:
            n_samples = n_train_stage2 if split=='train' else n_val_stage2
            return os.path.join(args.output_dir, 
                               f"{split}_stage2_N{N}_K{K}_n{n_samples}_"
                               f"mc{n_theta_samples}_erm{n_erm_samples}_seed{42 if split=='train' else 123}.pkl")
    
    # ========== STAGE 1: TRAINING DATA ==========
    train_stage1_path = get_cache_path("train", 1)
    if not args.force and os.path.exists(train_stage1_path):
        print(f"\nStage 1 training data exists: {train_stage1_path}")
        print("Loading...")
        with open(train_stage1_path, 'rb') as f:
            train_stage1_data = pickle.load(f)
    else:
        train_stage1_data = generator.generate_stage1_dataset(
            n_train_stage1, seed=42, desc="Train Stage 1"
        )
        if validate_dataset(train_stage1_data, "train", 1):
            with open(train_stage1_path, 'wb') as f:
                pickle.dump(train_stage1_data, f)
            print(f"Saved: {train_stage1_path}")
    
    # ========== STAGE 1: VALIDATION DATA ==========
    val_stage1_path = get_cache_path("val", 1)
    if not args.force and os.path.exists(val_stage1_path):
        print(f"\nStage 1 validation data exists: {val_stage1_path}")
        print("Loading...")
        with open(val_stage1_path, 'rb') as f:
            val_stage1_data = pickle.load(f)
    else:
        val_stage1_data = generator.generate_stage1_dataset(
            n_val_stage1, seed=123, desc="Val Stage 1"
        )
        if validate_dataset(val_stage1_data, "val", 1):
            with open(val_stage1_path, 'wb') as f:
                pickle.dump(val_stage1_data, f)
            print(f"Saved: {val_stage1_path}")
    
    # ========== STAGE 2: TRAINING DATA ==========
    train_stage2_path = get_cache_path("train", 2)
    if not args.force and os.path.exists(train_stage2_path):
        print(f"\nStage 2 training data exists: {train_stage2_path}")
    else:
        train_stage2_data = generator.generate_stage2_dataset(
            train_stage1_data, n_train_stage2, seed=42, desc="Train Stage 2"
        )
        if validate_dataset(train_stage2_data, "train", 2):
            with open(train_stage2_path, 'wb') as f:
                pickle.dump(train_stage2_data, f)
            print(f"Saved: {train_stage2_path}")
    
    # ========== STAGE 2: VALIDATION DATA ==========
    val_stage2_path = get_cache_path("val", 2)
    if not args.force and os.path.exists(val_stage2_path):
        print(f"\nStage 2 validation data exists: {val_stage2_path}")
    else:
        val_stage2_data = generator.generate_stage2_dataset(
            val_stage1_data, n_val_stage2, seed=123, desc="Val Stage 2"
        )
        if validate_dataset(val_stage2_data, "val", 2):
            with open(val_stage2_path, 'wb') as f:
                pickle.dump(val_stage2_data, f)
            print(f"Saved: {val_stage2_path}")
    
    print("\n" + "="*80)
    print("DATA GENERATION COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print(f"  {train_stage1_path}")
    print(f"  {val_stage1_path}")
    print(f"  {train_stage2_path}")
    print(f"  {val_stage2_path}")
    print("\nNext step:")
    print(f"  python train.py --mode dad_with_surrogate --config {args.config}")
    print("="*80)
    return 0


if __name__ == "__main__":
    exit(main())