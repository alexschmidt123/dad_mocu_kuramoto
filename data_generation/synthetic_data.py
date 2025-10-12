"""
Fixed synthetic data generation for training the MPNN surrogate model.
Properly computes MOCU, ERM, and Sync prediction tasks with Monte Carlo estimation.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from core.belief import init_history, update_intervals, pair_threshold, build_belief_graph, History
from core.kuramoto_env import PairTestEnv
from core.pacemaker_control import simulate_with_pacemaker, sync_check
from core.bisection import find_min_a_ctrl
import itertools


class SyntheticDataGenerator:
    """Generates synthetic training data for the MPNN surrogate model."""
    
    def __init__(self, N: int = 5, K: int = 4, prior_bounds: Tuple[float, float] = (0.05, 0.50),
                 omega_range: Tuple[float, float] = (-1.0, 1.0), n_samples: int = 1000,
                 sim_opts: Optional[Dict] = None, n_theta_samples: int = 20):
        self.N = N
        self.K = K
        self.prior_bounds = prior_bounds
        self.omega_range = omega_range
        self.n_samples = n_samples
        self.n_theta_samples = n_theta_samples  # For MC estimation
        self.sim_opts = sim_opts or {
            'dt': 0.01, 'T': 5.0, 'burn_in': 2.0, 'R_target': 0.95, 'method': 'RK45', 'n_trajectories': 3
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
    
    def sample_theta_from_belief(self, h: History, rng: np.random.Generator) -> np.ndarray:
        """Sample a coupling matrix from the current belief (uniform on intervals)."""
        A = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i+1, self.N):
                # Sample uniformly from [lower, upper]
                a_ij = rng.uniform(h.lower[i, j], h.upper[i, j])
                A[i, j] = a_ij
                A[j, i] = a_ij
        return A
    
    def compute_control_cost(self, a_ctrl: float, A: np.ndarray, omega: np.ndarray) -> float:
        """
        Compute control cost: C(a_ctrl, θ).
        Here we use C(a) = a if system syncs, else infinity.
        This makes a* = minimal a_ctrl that achieves sync.
        """
        try:
            if sync_check(A, omega, a_ctrl, **self.sim_opts):
                return a_ctrl
            else:
                return float('inf')
        except:
            return float('inf')
    
    def find_optimal_control(self, A: np.ndarray, omega: np.ndarray) -> float:
        """Find a*(θ) = minimal a_ctrl that synchronizes A."""
        def check_fn(a_ctrl):
            return sync_check(A, omega, a_ctrl, **self.sim_opts)
        
        try:
            return find_min_a_ctrl(A, omega, check_fn, lo=0.0, hi_init=0.1, tol=1e-3)
        except:
            return 2.0  # Fallback if search fails
    
    def compute_mocu(self, h: History, omega: np.ndarray, rng: np.random.Generator) -> float:
        """
        Compute MOCU properly:
        MOCU(h) = E_θ|h[C(a_IBR(h), θ) - C(a*(θ), θ)]
        
        Where a_IBR(h) is the Bayes-optimal control given belief h.
        For simplicity, we approximate a_IBR as the control that works for worst-case A_min.
        """
        # Compute worst-case control (IBR approximation)
        A_min = h.lower.copy()
        a_ibr = self.find_optimal_control(A_min, omega)
        
        # Monte Carlo estimate of MOCU
        mocu_sum = 0.0
        n_valid = 0
        
        for _ in range(self.n_theta_samples):
            # Sample θ from belief
            A_sample = self.sample_theta_from_belief(h, rng)
            
            # Compute a*(θ) for this sample
            a_star = self.find_optimal_control(A_sample, omega)
            
            # Compute costs
            cost_ibr = self.compute_control_cost(a_ibr, A_sample, omega)
            cost_star = self.compute_control_cost(a_star, A_sample, omega)
            
            # MOCU contribution (handle inf carefully)
            if cost_ibr < float('inf') and cost_star < float('inf'):
                mocu_sum += (cost_ibr - cost_star)
                n_valid += 1
        
        if n_valid > 0:
            return mocu_sum / n_valid
        else:
            # If no valid samples, use a_ibr as penalty
            return a_ibr * 0.5
    
    def compute_erm(self, h: History, xi: Tuple[int, int], omega: np.ndarray, 
                    rng: np.random.Generator) -> float:
        """
        Compute ERM properly:
        ERM(h, ξ) = E_y|h,ξ[MOCU(h ⊕ (ξ,y))]
        
        This requires computing expected MOCU over both possible outcomes.
        """
        i, j = xi
        lam = pair_threshold(omega, i, j)
        
        # Compute p(sync | h, ξ) using current belief intervals
        lo, up = h.lower[i, j], h.upper[i, j]
        
        # Clamp lambda to interval
        lam_clamped = max(lo, min(lam, up))
        
        # Probability of sync (uniform prior on [lo, up])
        if up > lo:
            p_sync = max(0.0, (up - lam_clamped) / (up - lo))
        else:
            p_sync = 1.0 if lo >= lam else 0.0
        
        p_not_sync = 1.0 - p_sync
        
        # Compute MOCU for both outcomes
        erm = 0.0
        
        if p_sync > 1e-6:
            # Create hypothetical belief after observing sync
            h_sync = self._copy_history(h)
            update_intervals(h_sync, xi, True, omega)
            mocu_sync = self.compute_mocu(h_sync, omega, rng)
            erm += p_sync * mocu_sync
        
        if p_not_sync > 1e-6:
            # Create hypothetical belief after observing not-sync
            h_not = self._copy_history(h)
            update_intervals(h_not, xi, False, omega)
            mocu_not = self.compute_mocu(h_not, omega, rng)
            erm += p_not_sync * mocu_not
        
        return erm
    
    def run_experiment_sequence(self, omega: np.ndarray, A_true: np.ndarray, 
                               rng: np.random.Generator) -> Dict:
        """Run a complete experiment sequence and collect data."""
        # Initialize history
        h = init_history(self.N, self.prior_bounds)
        
        experiment_data = []
        
        for step in range(self.K):
            # Get current belief graph
            belief_graph = build_belief_graph(h, omega)
            
            # Compute MOCU at this step
            current_mocu = self.compute_mocu(h, omega, rng)
            
            # Get all candidate pairs
            candidates = [(i, j) for i in range(self.N) for j in range(i+1, self.N)]
            
            # Compute ERM for each candidate
            erm_scores = {}
            for cand_xi in candidates:
                erm_scores[cand_xi] = self.compute_erm(h, cand_xi, omega, rng)
            
            # Choose random pair for this training trajectory
            xi = candidates[rng.integers(len(candidates))]
            
            # Simulate true outcome
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
        
        # Final belief and labels
        final_belief_graph = build_belief_graph(h, omega)
        final_mocu = self.compute_mocu(h, omega, rng)
        
        # Compute a_ctrl_star for final state
        A_min = h.lower.copy()
        a_ctrl_star = self.find_optimal_control(A_min, omega)
        
        # Compute sync labels for various a_ctrl values
        sync_scores = {}
        a_ctrl_values = np.linspace(0.0, 1.0, 10)
        for a_ctrl in a_ctrl_values:
            try:
                sync_scores[float(a_ctrl)] = float(sync_check(A_min, omega, a_ctrl, **self.sim_opts))
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
    
    def _copy_history(self, h: History) -> History:
        """Create a deep copy of history."""
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
        print(f"Using {self.n_theta_samples} samples for MOCU/ERM estimation")
        
        for i in range(self.n_samples):
            if (i + 1) % 50 == 0:
                print(f"Generated {i + 1}/{self.n_samples} samples")
            
            # Generate random parameters
            omega = self.generate_omega(rng)
            A_true = self.generate_true_A(rng)
            
            # Run experiment sequence
            try:
                data = self.run_experiment_sequence(omega, A_true, rng)
                dataset.append(data)
            except Exception as e:
                print(f"Warning: Failed to generate sample {i}: {e}")
                continue
        
        print(f"Data generation complete! Generated {len(dataset)} valid samples.")
        return dataset


def create_training_data(N: int = 5, K: int = 4, n_samples: int = 1000, 
                        n_theta_samples: int = 20, seed: int = 42) -> List[Dict]:
    """Convenience function to create training data."""
    generator = SyntheticDataGenerator(
        N=N, K=K, n_samples=n_samples, n_theta_samples=n_theta_samples
    )
    return generator.generate_dataset(seed=seed)


if __name__ == "__main__":
    # Generate a small dataset for testing
    print("Testing fixed data generation...")
    dataset = create_training_data(N=5, K=4, n_samples=10, n_theta_samples=10, seed=42)
    
    if dataset:
        print(f"\n✓ Generated {len(dataset)} samples")
        sample = dataset[0]
        print(f"✓ Sample keys: {list(sample.keys())}")
        print(f"✓ Experiment data length: {len(sample['experiment_data'])}")
        print(f"✓ Final MOCU: {sample['final_mocu']:.4f}")
        print(f"✓ a_ctrl_star: {sample['a_ctrl_star']:.4f}")
        print(f"✓ First step MOCU: {sample['experiment_data'][0]['mocu']:.4f}")
        print(f"✓ Number of ERM scores per step: {len(sample['experiment_data'][0]['erm_scores'])}")