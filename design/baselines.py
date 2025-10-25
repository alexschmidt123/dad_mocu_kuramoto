#!/usr/bin/env python3
"""
CORRECTED Baselines matching AccelerateOED 2023 Paper

Paper's Methods:
1. Random - Random pair selection
2. Entropy - Max uncertainty (max upper-lower width)
3. ODE (non-iterative) - One-shot MOCU at start
4. iODE (iterative) - Recompute MOCU after each step
5. MP (non-iterative) - MPNN surrogate, one-shot
6. iMP (iterative) - MPNN surrogate, recompute each step
7. DAD (your innovation) - Policy network

Key differences:
- "Fixed Design" in DAD papers = "ODE non-iterative" in paper2023
- Paper2023's innovation was MPNN acceleration (MP/iMP)
- Your innovation is DAD policy on top of MPNN
"""

import numpy as np
from typing import List, Tuple
from core.belief import pair_threshold


# ==============================================================================
# BASELINE 1: Random
# ==============================================================================

def random_chooser(env, candidates):
    """Random baseline - uniformly sample untested pairs"""
    if not candidates:
        return (0, 1)
    return candidates[np.random.randint(len(candidates))]


# ==============================================================================
# BASELINE 2: Entropy (Max Uncertainty)
# ==============================================================================

def entropy_chooser(env, candidates):
    """
    Entropy baseline - select pair with maximum uncertainty.
    
    Paper: "select the pair with the largest interval width"
    This is equivalent to maximum information gain heuristic.
    """
    if not candidates:
        return (0, 1)
    
    max_width = -1
    best_pair = candidates[0]
    
    for (i, j) in candidates:
        width = env.h.upper[i, j] - env.h.lower[i, j]
        if width > max_width:
            max_width = width
            best_pair = (i, j)
    
    return best_pair


# ==============================================================================
# BASELINE 3: ODE Non-iterative (Paper's "Fixed Design")
# ==============================================================================

def ode_noniterative_chooser_factory(precomputed_sequence: List[Tuple[int, int]]):
    """
    ODE non-iterative - compute optimal sequence once at start.
    
    Paper approach:
    1. At t=0, compute ERM for all pairs using ground-truth ODE
    2. Select K pairs greedily (min ERM at each step)
    3. Use this fixed sequence for all problem instances
    
    This is what DAD papers call "Fixed Design" baseline.
    """
    step_counter = [0]
    
    def ode_chooser(env, candidates):
        if step_counter[0] < len(precomputed_sequence):
            chosen = precomputed_sequence[step_counter[0]]
            step_counter[0] += 1
            if chosen in candidates:
                return chosen
        return candidates[0] if candidates else (0, 1)
    
    return ode_chooser


def compute_ode_noniterative_sequence(env, n_steps: int) -> List[Tuple[int, int]]:
    """
    Compute fixed sequence using one-shot ERM with ground-truth ODE.
    
    This is expensive - only done once offline for "Fixed Design" baseline.
    """
    from core.pacemaker_control import sync_check
    from core.bisection import find_min_a_ctrl
    import copy
    
    sequence = []
    sim_opts = {'dt': 0.01, 'T': 5.0, 'burn_in': 2.0, 'R_target': 0.95, 
                'method': 'RK45', 'n_trajectories': 3}
    
    # Initial state
    h_current = copy.deepcopy(env.h)
    
    print("Computing ODE non-iterative (Fixed Design) sequence...")
    for step in range(n_steps):
        candidates = [(i, j) for i in range(env.N) for j in range(i+1, env.N)
                     if not h_current.tested[i, j]]
        
        if not candidates:
            break
        
        print(f"  Step {step+1}/{n_steps}: Evaluating {len(candidates)} candidates...")
        
        # Compute ERM for each candidate using ground-truth ODE
        best_pair = None
        best_erm = float('inf')
        
        for xi in candidates:
            i, j = xi
            lam = pair_threshold(env.omega, i, j)
            
            # Estimate ERM by sampling
            erm_samples = []
            for _ in range(5):  # Few samples for speed
                # Sample from belief
                A_sample = sample_from_belief(h_current, env.N, env.rng)
                y_sync = (A_sample[i, j] >= lam)
                
                # Posterior
                h_post = copy.deepcopy(h_current)
                from core.belief import update_intervals
                update_intervals(h_post, xi, y_sync, env.omega)
                
                # Compute posterior MOCU using ground-truth ODE
                mocu_post = compute_mocu_ode(h_post, env.omega, sim_opts, K_samples=100)
                erm_samples.append(mocu_post)
            
            erm = np.mean(erm_samples)
            
            if erm < best_erm:
                best_erm = erm
                best_pair = xi
        
        sequence.append(best_pair)
        
        # Update belief (using true outcome for this env)
        i, j = best_pair
        lam = pair_threshold(env.omega, i, j)
        y_sync = (env.A_true[i, j] >= lam)
        from core.belief import update_intervals
        update_intervals(h_current, best_pair, y_sync, env.omega)
        
        print(f"    Selected: {best_pair}, ERM: {best_erm:.4f}")
    
    return sequence


def sample_from_belief(h, N, rng):
    """Sample coupling matrix from belief"""
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            lower = h.lower[i, j]
            upper = h.upper[i, j]
            if upper <= lower:
                a_ij = (lower + upper) / 2 if upper > 0 else 0.1
            else:
                a_ij = rng.uniform(lower, upper)
            A[i, j] = a_ij
            A[j, i] = a_ij
    return A


def compute_mocu_ode(h, omega, sim_opts, K_samples=100):
    """Compute MOCU using ground-truth ODE (expensive)"""
    from core.pacemaker_control import sync_check
    from core.bisection import find_min_a_ctrl
    
    rng = np.random.default_rng()
    a_optimal_samples = []
    
    for _ in range(K_samples):
        A_sample = sample_from_belief(h, len(omega), rng)
        
        def check_fn(a_ctrl):
            try:
                return sync_check(A_sample, omega, a_ctrl, **sim_opts)
            except:
                return False
        
        try:
            a_opt = find_min_a_ctrl(A_sample, omega, check_fn, tol=0.005, verbose=False)
            a_optimal_samples.append(a_opt)
        except:
            a_optimal_samples.append(2.0)
    
    if not a_optimal_samples:
        return 0.1
    
    a_IBR = np.max(a_optimal_samples)
    mocu = a_IBR - np.mean(a_optimal_samples)
    return max(0.0, mocu)


# ==============================================================================
# BASELINE 4: iODE Iterative
# ==============================================================================

def iode_iterative_chooser(env, candidates):
    """
    iODE iterative - recompute ERM using ground-truth ODE at each step.
    
    Paper approach:
    - At each step, compute ERM for all candidates using ODE
    - Select pair with minimum ERM
    - Very expensive but most accurate without surrogate
    """
    if not candidates:
        return (0, 1)
    
    print(f"  iODE: Evaluating {len(candidates)} candidates...")
    
    sim_opts = {'dt': 0.01, 'T': 5.0, 'burn_in': 2.0, 'R_target': 0.95, 
                'method': 'RK45', 'n_trajectories': 3}
    
    best_pair = None
    best_erm = float('inf')
    
    for xi in candidates:
        i, j = xi
        lam = pair_threshold(env.omega, i, j)
        
        # Estimate ERM
        erm_samples = []
        for _ in range(5):  # Few samples for speed
            A_sample = sample_from_belief(env.h, env.N, env.rng)
            y_sync = (A_sample[i, j] >= lam)
            
            import copy
            from core.belief import update_intervals
            h_post = copy.deepcopy(env.h)
            update_intervals(h_post, xi, y_sync, env.omega)
            
            mocu_post = compute_mocu_ode(h_post, env.omega, sim_opts, K_samples=50)
            erm_samples.append(mocu_post)
        
        erm = np.mean(erm_samples)
        
        if erm < best_erm:
            best_erm = erm
            best_pair = xi
    
    print(f"    Selected: {best_pair}, ERM: {best_erm:.4f}")
    return best_pair


# ==============================================================================
# BASELINE 5: MP Non-iterative (MPNN, one-shot)
# ==============================================================================

def mp_noniterative_chooser_factory(surrogate, precomputed_sequence: List[Tuple[int, int]]):
    """
    MP non-iterative - compute sequence once using MPNN surrogate.
    
    Paper's innovation: Use MPNN to approximate ERM instead of expensive ODE.
    - Compute sequence once at start using MPNN
    - Much faster than ODE but still one-shot
    """
    step_counter = [0]
    
    def mp_chooser(env, candidates):
        if step_counter[0] < len(precomputed_sequence):
            chosen = precomputed_sequence[step_counter[0]]
            step_counter[0] += 1
            if chosen in candidates:
                return chosen
        return candidates[0] if candidates else (0, 1)
    
    return mp_chooser


def compute_mp_noniterative_sequence(env, surrogate, n_steps: int) -> List[Tuple[int, int]]:
    """Compute fixed sequence using one-shot MPNN surrogate."""
    import copy
    from core.belief import update_intervals
    
    sequence = []
    h_current = copy.deepcopy(env.h)
    
    print("Computing MP non-iterative (MPNN Fixed) sequence...")
    for step in range(n_steps):
        candidates = [(i, j) for i in range(env.N) for j in range(i+1, env.N)
                     if not h_current.tested[i, j]]
        
        if not candidates:
            break
        
        # Use MPNN to compute ERM
        from design.greedy_erm import choose_next_pair_greedy
        env_temp = copy.deepcopy(env)
        env_temp.h = h_current
        
        best_pair = choose_next_pair_greedy(env_temp, candidates)
        sequence.append(best_pair)
        
        # Update belief
        i, j = best_pair
        lam = pair_threshold(env.omega, i, j)
        y_sync = (env.A_true[i, j] >= lam)
        update_intervals(h_current, best_pair, y_sync, env.omega)
        
        print(f"  Step {step+1}/{n_steps}: Selected {best_pair}")
    
    return sequence


# ==============================================================================
# BASELINE 6: iMP Iterative (MPNN, adaptive)
# ==============================================================================

def imp_iterative_chooser(env, candidates):
    """
    iMP iterative - recompute ERM using MPNN at each step.
    
    Paper's AccelerateOED method (adaptive):
    - At each step, use MPNN to compute ERM for all candidates
    - Select pair with minimum ERM
    - Fast and adaptive
    
    This is your "Greedy MPNN" baseline.
    """
    from design.greedy_erm import choose_next_pair_greedy
    return choose_next_pair_greedy(env, candidates)


# ==============================================================================
# Summary and Usage
# ==============================================================================

def get_all_baselines(surrogate=None):
    """
    Get all baseline methods matching paper2023.
    
    Returns dict of {name: chooser_fn}
    """
    baselines = {
        'Random': random_chooser,
        'Entropy (Max Uncertainty)': entropy_chooser,
        # ODE methods require precomputation
        # 'ODE (non-iterative)': requires env and precomputation
        # 'iODE (iterative)': iode_iterative_chooser,  # Very slow
    }
    
    if surrogate is not None:
        # MP methods available with surrogate
        baselines['Greedy MPNN (iMP)'] = imp_iterative_chooser
        # 'MP (non-iterative)': requires precomputation
    
    return baselines


if __name__ == "__main__":
    print("="*80)
    print("CORRECTED BASELINES (AccelerateOED 2023 Paper)")
    print("="*80)
    print("\nPaper's Methods:")
    print("1. Random - Random pair selection")
    print("2. Entropy - Max uncertainty (max interval width)")
    print("3. ODE (non-iterative) - One-shot ground-truth MOCU")
    print("4. iODE (iterative) - Recompute ground-truth MOCU each step [SLOW]")
    print("5. MP (non-iterative) - One-shot MPNN surrogate")
    print("6. iMP (iterative) - Adaptive MPNN surrogate [Paper's innovation]")
    print("7. DAD Policy - Your innovation (policy network with MPNN)")
    print("\nKey Insights:")
    print("- 'Fixed Design' in DAD = 'ODE/MP non-iterative' in paper2023")
    print("- Paper2023's innovation: MPNN acceleration (orders of magnitude faster)")
    print("- Your innovation: DAD policy learns better than greedy MPNN")
    print("="*80)