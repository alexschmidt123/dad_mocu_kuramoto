"""
CORRECTED data_generation/synthetic_data.py - Generate complete data with MOCU + ERM + Sync.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from core.belief import init_history, update_intervals, pair_threshold, build_belief_graph, History
from core.pacemaker_control import sync_check
from core.bisection import find_min_a_ctrl


class SyntheticDataGenerator:
    """Generate complete training data with MOCU, ERM, and Sync labels."""
    
    def __init__(self, N: int = 5, K: int = 4, prior_bounds: Tuple[float, float] = (0.05, 0.50),
                 omega_range: Tuple[float, float] = (-1.0, 1.0), n_samples: int = 1000,
                 sim_opts: Optional[Dict] = None, n_theta_samples: int = 20):
        self.N = N
        self.K = K
        self.prior_bounds = prior_bounds
        self.omega_range = omega_range
        self.n_samples = n_samples
        self.n_theta_samples = n_theta_samples
        self.sim_opts = sim_opts or {
            'dt': 0.01, 'T': 5.0, 'burn_in': 2.0, 'R_target': 0.95, 'method': 'RK45', 'n_trajectories': 3
        }
        
        # Enable complete data generation
        self.compute_true_erm = True
        self.n_erm_samples = 10
        
    def generate_omega(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.omega_range[0], self.omega_range[1], size=self.N)
    
    def generate_true_A(self, rng: np.random.Generator) -> np.ndarray:
        lo, hi = self.prior_bounds
        A = rng.uniform(lo, hi, size=(self.N, self.N))
        A = 0.5 * (A + A.T)
        np.fill_diagonal(A, 0.0)
        return A
    
    def sample_theta_from_belief(self, h: History, rng: np.random.Generator) -> np.ndarray:
        A = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i+1, self.N):
                lower_bound = max(h.lower[i, j], self.prior_bounds[0])
                upper_bound = min(h.upper[i, j], self.prior_bounds[1])
                
                if upper_bound <= lower_bound:
                    a_ij = rng.uniform(self.prior_bounds[0], self.prior_bounds[1])
                else:
                    a_ij = rng.uniform(lower_bound, upper_bound)
                
                A[i, j] = a_ij
                A[j, i] = a_ij
        return A
    
    def find_optimal_control(self, A: np.ndarray, omega: np.ndarray) -> float:
        """Find a*(theta) = minimal a_ctrl that synchronizes A."""
        def check_fn(a_ctrl):
            try:
                return sync_check(A, omega, a_ctrl, **self.sim_opts)
            except:
                return False
        
        try:
            return find_min_a_ctrl(A, omega, check_fn, lo=0.0, hi_init=0.1, 
                                  tol=5e-3, verbose=False)
        except:
            return 2.0
    
    def compute_mocu(self, h: History, omega: np.ndarray, rng: np.random.Generator) -> float:
        """Compute MOCU = a*(A_min) - E[a*(theta)]"""
        try:
            # Worst-case control
            A_min = h.lower.copy()
            np.fill_diagonal(A_min, 0.0)
            a_worst_case = self.find_optimal_control(A_min, omega)
            
            # Expected optimal control
            optimal_controls = []
            for _ in range(self.n_theta_samples):
                A_sample = self.sample_theta_from_belief(h, rng)
                a_optimal = self.find_optimal_control(A_sample, omega)
                optimal_controls.append(a_optimal)
            
            expected_optimal = np.mean(optimal_controls) if optimal_controls else a_worst_case
            return max(0.0, min(5.0, a_worst_case - expected_optimal))
        except:
            return 0.1
    
    def compute_erm(self, h: History, xi: Tuple[int, int], omega: np.ndarray, 
                    rng: np.random.Generator) -> float:
        """Compute Expected Remaining MOCU for experiment xi."""
        import copy
        
        try:
            i, j = xi
            lam = pair_threshold(omega, i, j)
            erm_values = []
            
            for _ in range(self.n_erm_samples):
                # Sample theta from current belief
                A_sample = self.sample_theta_from_belief(h, rng)
                
                # Determine outcome
                y_sync = (A_sample[i, j] >= lam)
                
                # Create posterior belief
                h_posterior = copy.deepcopy(h)
                update_intervals(h_posterior, xi, y_sync, omega)
                
                # Compute MOCU after this experiment
                posterior_mocu = self.compute_mocu(h_posterior, omega, rng)
                erm_values.append(posterior_mocu)
            
            return max(0.0, np.mean(erm_values)) if erm_values else 0.0
        except:
            return 0.0
    
    def compute_all_erm_scores(self, h: History, candidates: List[Tuple[int, int]], 
                               omega: np.ndarray, rng: np.random.Generator,
                               max_candidates: int = 10) -> Dict[Tuple[int, int], float]:
        """Compute ERM for all (or subset of) candidate experiments."""
        erm_scores = {}
        
        # Sample candidates if too many
        if len(candidates) > max_candidates:
            selected = rng.choice(len(candidates), size=min(max_candidates, len(candidates)), replace=False)
            candidates_to_eval = [candidates[i] for i in selected]
        else:
            candidates_to_eval = candidates
        
        for xi in candidates_to_eval:
            erm = self.compute_erm(h, xi, omega, rng)
            erm_scores[xi] = erm
        
        return erm_scores
    
    def compute_sync_scores(self, A_min: np.ndarray, omega: np.ndarray) -> Dict[float, int]:
        """Compute synchronization labels for different control values."""
        sync_scores = {}
        
        # Test a few control values
        test_controls = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
        
        for a_ctrl in test_controls:
            try:
                is_sync = sync_check(A_min, omega, a_ctrl, **self.sim_opts)
                sync_scores[a_ctrl] = 1 if is_sync else 0
            except:
                sync_scores[a_ctrl] = 0
        
        return sync_scores
    
    def run_experiment_sequence(self, omega: np.ndarray, A_true: np.ndarray, 
                               rng: np.random.Generator) -> Dict:
        """Run complete K-step experiment sequence with all labels."""
        h = init_history(self.N, self.prior_bounds)
        experiment_data = []
        
        for step in range(self.K):
            # Current belief state
            belief_graph = build_belief_graph(h, omega)
            current_mocu = self.compute_mocu(h, omega, rng)
            
            # Get candidate pairs (untested pairs)
            candidates = [(i, j) for i in range(self.N) for j in range(i+1, self.N)
                         if not h.tested[i, j]]
            
            # If all pairs tested, allow retesting
            if not candidates:
                candidates = [(i, j) for i in range(self.N) for j in range(i+1, self.N)]
            
            # Compute ERM scores for all candidates
            erm_scores = self.compute_all_erm_scores(h, candidates, omega, rng, max_candidates=10)
            
            # Random experiment selection for training data diversity
            xi = candidates[rng.integers(len(candidates))]
            i, j = xi
            lam = pair_threshold(omega, i, j)
            y_sync = (A_true[i, j] >= lam)
            
            # Store step data
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
            
            # Update belief
            update_intervals(h, xi, y_sync, omega)
        
        # Final state
        final_belief_graph = build_belief_graph(h, omega)
        final_mocu = self.compute_mocu(h, omega, rng)
        
        # Compute final control parameter
        A_min = h.lower.copy()
        np.fill_diagonal(A_min, 0.0)
        a_ctrl_star = self.find_optimal_control(A_min, omega)
        
        # Compute sync scores for training
        sync_scores = self.compute_sync_scores(A_min, omega)
        
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
    
    def generate_dataset(self, seed: int = 42) -> List[Dict]:
        """Generate complete training dataset with MOCU + ERM + Sync labels."""
        rng = np.random.default_rng(seed)
        dataset = []
        
        print(f"Generating {self.n_samples} samples with complete labels (MOCU + ERM + Sync)")
        print(f"Using {self.n_theta_samples} MC samples for MOCU estimation")
        print(f"Using {self.n_erm_samples} MC samples for ERM estimation")
        
        successful_samples = 0
        failed_samples = 0
        
        for i in range(self.n_samples):
            if (i + 1) % 50 == 0:
                print(f"  Generated {i + 1}/{self.n_samples} samples "
                      f"(Success: {successful_samples}, Failed: {failed_samples})")
            
            try:
                omega = self.generate_omega(rng)
                A_true = self.generate_true_A(rng)
                
                data = self.run_experiment_sequence(omega, A_true, rng)
                dataset.append(data)
                successful_samples += 1
                
            except Exception as e:
                failed_samples += 1
                print(f"  Warning: Failed sample {i}: {e}")
                continue
        
        print(f"\nSUCCESS: Generated {len(dataset)} valid samples")
        print(f"Success rate: {successful_samples}/{self.n_samples} ({100*successful_samples/self.n_samples:.1f}%)")
        
        # Verify label statistics
        if dataset:
            all_mocu = []
            all_erm = []
            all_sync = []
            
            for sample in dataset[:min(50, len(dataset))]:
                for step_data in sample['experiment_data']:
                    all_mocu.append(step_data['mocu'])
                    all_erm.extend(list(step_data['erm_scores'].values()))
                all_mocu.append(sample['final_mocu'])
                all_sync.extend(list(sample['sync_scores'].values()))
            
            all_mocu = np.array(all_mocu)
            
            print(f"\nLabel Statistics:")
            print(f"  MOCU: {len(all_mocu)} labels, mean={all_mocu.mean():.4f}, range=[{all_mocu.min():.4f}, {all_mocu.max():.4f}]")
            if all_erm:
                all_erm = np.array(all_erm)
                print(f"  ERM: {len(all_erm)} labels, mean={all_erm.mean():.4f}, range=[{all_erm.min():.4f}, {all_erm.max():.4f}]")
            if all_sync:
                all_sync = np.array(all_sync)
                print(f"  Sync: {len(all_sync)} labels, mean={all_sync.mean():.4f}, unique values={np.unique(all_sync)}")
            
            if all_mocu.min() < 0:
                print(f"  WARNING: Negative MOCU detected!")
                return []
            else:
                print(f"  SUCCESS: All labels generated successfully")
        
        return dataset


def create_training_data(N: int = 5, K: int = 4, n_samples: int = 1000, 
                        n_theta_samples: int = 20, seed: int = 42) -> List[Dict]:
    """Convenience function to generate complete training data."""
    generator = SyntheticDataGenerator(
        N=N, K=K, n_samples=n_samples, n_theta_samples=n_theta_samples
    )
    return generator.generate_dataset(seed=seed)


if __name__ == "__main__":
    print("Testing complete data generation...")
    dataset = create_training_data(N=5, K=4, n_samples=5, n_theta_samples=10, seed=42)
    
    if dataset:
        print(f"\nSUCCESS: Generated {len(dataset)} samples")
        sample = dataset[0]
        print(f"Initial MOCU: {sample['experiment_data'][0]['mocu']:.4f}")
        print(f"Final MOCU: {sample['final_mocu']:.4f}")
        print(f"a_ctrl_star: {sample['a_ctrl_star']:.4f}")
        print(f"ERM scores in first step: {len(sample['experiment_data'][0]['erm_scores'])}")
        print(f"Sync scores: {len(sample['sync_scores'])}")
        print(f"Available keys: {list(sample.keys())}")
    else:
        print("FAILED: No samples generated")