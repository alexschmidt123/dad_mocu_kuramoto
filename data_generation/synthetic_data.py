"""
Synthetic data generation for training the MPNN surrogate model.
Generates training data for MOCU, ERM, and Sync prediction tasks.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from core.belief import init_history, update_intervals, pair_threshold, build_belief_graph
from core.kuramoto_env import PairTestEnv
from core.pacemaker_control import simulate_with_pacemaker, sync_check
from core.bisection import find_min_a_ctrl
import itertools


class SyntheticDataGenerator:
    """Generates synthetic training data for the MPNN surrogate model."""
    
    def __init__(self, N: int = 5, K: int = 4, prior_bounds: Tuple[float, float] = (0.05, 0.50),
                 omega_range: Tuple[float, float] = (-1.0, 1.0), n_samples: int = 1000,
                 sim_opts: Optional[Dict] = None):
        self.N = N
        self.K = K
        self.prior_bounds = prior_bounds
        self.omega_range = omega_range
        self.n_samples = n_samples
        self.sim_opts = sim_opts or {
            'dt': 0.01, 'T': 5.0, 'burn_in': 2.0, 'R_target': 0.95
        }
        
    def generate_omega(self, rng: np.random.Generator) -> np.ndarray:
        """Generate natural frequencies for oscillators."""
        return rng.uniform(self.omega_range[0], self.omega_range[1], size=self.N)
    
    def generate_true_A(self, rng: np.random.Generator) -> np.ndarray:
        """Generate true coupling matrix A."""
        lo, hi = self.prior_bounds
        A = rng.uniform(lo, hi, size=(self.N, self.N))
        A = 0.5 * (A + A.T)  # Make symmetric
        np.fill_diagonal(A, 0.0)  # Zero diagonal
        return A
    
    def run_experiment_sequence(self, omega: np.ndarray, A_true: np.ndarray, 
                               rng: np.random.Generator) -> Dict:
        """Run a complete experiment sequence and collect data."""
        # Initialize environment
        env = PairTestEnv(self.N, omega, self.prior_bounds, self.K, rng=rng)
        env.A_true = A_true  # Set the true matrix
        
        # Run random experiment sequence
        experiment_data = []
        for step in range(self.K):
            # Get current belief graph
            belief_graph = env.features()
            
            # Choose random pair
            candidates = env.candidate_pairs()
            xi = candidates[rng.integers(len(candidates))]
            
            # Run experiment
            result = env.step(xi)
            
            # Store data for this step
            step_data = {
                'belief_graph': belief_graph,
                'candidate_pairs': candidates,
                'chosen_pair': xi,
                'outcome': result['y'],
                'step': step
            }
            experiment_data.append(step_data)
        
        # Final belief graph after all experiments
        final_belief_graph = env.features()
        
        # Compute ground truth labels
        labels = self._compute_ground_truth_labels(env, A_true, omega)
        
        return {
            'experiment_data': experiment_data,
            'final_belief_graph': final_belief_graph,
            'labels': labels,
            'omega': omega,
            'A_true': A_true
        }
    
    def _compute_ground_truth_labels(self, env: PairTestEnv, A_true: np.ndarray, 
                                   omega: np.ndarray) -> Dict:
        """Compute ground truth labels for MOCU, ERM, and Sync prediction."""
        # Build worst-case matrix from lower bounds
        A_min = env.h.lower.copy()
        
        # Compute MOCU (simplified version)
        mocu = self._compute_mocu(A_true, A_min, omega)
        
        # Compute ERM for each candidate pair
        erm_scores = {}
        candidates = env.candidate_pairs()
        for xi in candidates:
            erm_scores[xi] = self._compute_erm(env, xi, A_true, omega)
        
        # Compute sync check for different a_ctrl values
        sync_scores = {}
        a_ctrl_values = np.linspace(0.0, 2.0, 20)
        for a_ctrl in a_ctrl_values:
            sync_scores[a_ctrl] = sync_check(A_min, omega, a_ctrl, **self.sim_opts)
        
        return {
            'mocu': mocu,
            'erm_scores': erm_scores,
            'sync_scores': sync_scores,
            'a_ctrl_star': self._find_optimal_a_ctrl(A_min, omega)
        }
    
    def _compute_mocu(self, A_true: np.ndarray, A_min: np.ndarray, omega: np.ndarray) -> float:
        """Compute MOCU (simplified version)."""
        # This is a simplified MOCU computation
        # In practice, this would involve more sophisticated optimization
        a_ctrl_true = self._find_optimal_a_ctrl(A_true, omega)
        a_ctrl_min = self._find_optimal_a_ctrl(A_min, omega)
        return abs(a_ctrl_true - a_ctrl_min)
    
    def _compute_erm(self, env: PairTestEnv, xi: Tuple[int, int], 
                    A_true: np.ndarray, omega: np.ndarray) -> float:
        """Compute ERM for a candidate pair (simplified)."""
        # Simulate the outcome
        i, j = xi
        lam = pair_threshold(omega, i, j)
        y_sync = A_true[i, j] >= lam
        
        # Create hypothetical updated belief
        h_hyp = self._copy_history(env.h)
        update_intervals(h_hyp, xi, y_sync, omega)
        
        # Compute MOCU after update
        A_min_hyp = h_hyp.lower.copy()
        mocu_after = self._compute_mocu(A_true, A_min_hyp, omega)
        
        return mocu_after
    
    def _find_optimal_a_ctrl(self, A: np.ndarray, omega: np.ndarray) -> float:
        """Find optimal control parameter using binary search."""
        def check_fn(a_ctrl):
            return sync_check(A, omega, a_ctrl, **self.sim_opts)
        
        return find_min_a_ctrl(A, omega, check_fn)
    
    def _copy_history(self, h):
        """Create a copy of history for hypothetical updates."""
        from core.belief import History
        return History(
            pairs=h.pairs.copy(),
            outcomes=h.outcomes.copy(),
            lower=h.lower.copy(),
            upper=h.upper.copy(),
            tested=h.tested.copy()
        )
    
    def generate_dataset(self, seed: int = 42) -> List[Dict]:
        """Generate complete training dataset."""
        rng = np.random.default_rng(seed)
        dataset = []
        
        print(f"Generating {self.n_samples} synthetic experiments...")
        
        for i in range(self.n_samples):
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{self.n_samples} samples")
            
            # Generate random parameters
            omega = self.generate_omega(rng)
            A_true = self.generate_true_A(rng)
            
            # Run experiment sequence
            data = self.run_experiment_sequence(omega, A_true, rng)
            dataset.append(data)
        
        print("Data generation complete!")
        return dataset


def create_training_data(N: int = 5, K: int = 4, n_samples: int = 1000, 
                        seed: int = 42) -> List[Dict]:
    """Convenience function to create training data."""
    generator = SyntheticDataGenerator(N=N, K=K, n_samples=n_samples)
    return generator.generate_dataset(seed=seed)


if __name__ == "__main__":
    # Generate a small dataset for testing
    dataset = create_training_data(N=5, K=4, n_samples=100, seed=42)
    print(f"Generated {len(dataset)} samples")
    print(f"Sample keys: {list(dataset[0].keys())}")
    print(f"Experiment data length: {len(dataset[0]['experiment_data'])}")
    print(f"Labels keys: {list(dataset[0]['labels'].keys())}")
