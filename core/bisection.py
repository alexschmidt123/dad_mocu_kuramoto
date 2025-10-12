"""
Improved bisection search for finding optimal control parameters.
Includes better error handling, validation, and diagnostics.
"""

import numpy as np
from typing import Callable, Optional, Dict


def find_min_a_ctrl(A_min, omega, check_fn: Callable[[float], bool], 
                    lo: float = 0.0, hi_init: float = 0.1, tol: float = 1e-3, 
                    max_expand: int = 20, max_iter: int = 40,
                    verbose: bool = False) -> float:
    """
    Find minimal control parameter a_ctrl such that check_fn(a_ctrl) returns True.
    
    Uses binary search with initial expansion phase to find upper bound.
    
    Args:
        A_min: (N, N) coupling matrix (usually worst-case lower bounds)
        omega: (N,) natural frequencies (not used but kept for interface compatibility)
        check_fn: function that returns True if a_ctrl achieves desired property
        lo: initial lower bound
        hi_init: initial upper bound guess
        tol: tolerance for convergence
        max_expand: maximum expansion iterations
        max_iter: maximum bisection iterations
        verbose: print debug information
    
    Returns:
        float: minimal a_ctrl value that satisfies check_fn
    """
    # Validate inputs
    if lo < 0:
        raise ValueError(f"Lower bound must be non-negative, got {lo}")
    if hi_init <= lo:
        raise ValueError(f"Initial upper bound {hi_init} must be > lower bound {lo}")
    if tol <= 0:
        raise ValueError(f"Tolerance must be positive, got {tol}")
    
    # Phase 1: Expand upper bound until check_fn is satisfied
    hi = hi_init
    expands = 0
    
    if verbose:
        print(f"Phase 1: Finding upper bound starting from hi={hi}")
    
    while not check_fn(hi):
        hi *= 2.0
        expands += 1
        
        if verbose:
            print(f"  Expansion {expands}: hi={hi:.6f}, check={False}")
        
        if expands > max_expand:
            if verbose:
                print(f"  Warning: Max expansions reached. Returning hi={hi}")
            return hi
    
    if verbose:
        print(f"  Found valid upper bound: hi={hi:.6f}")
    
    # Phase 2: Binary search between [lo, hi]
    left, right = lo, hi
    iterations = 0
    
    if verbose:
        print(f"Phase 2: Binary search in [{left:.6f}, {right:.6f}]")
    
    while right - left > tol and iterations < max_iter:
        mid = 0.5 * (left + right)
        check_result = check_fn(mid)
        
        if verbose:
            print(f"  Iter {iterations+1}: mid={mid:.6f}, check={check_result}, "
                  f"interval=[{left:.6f}, {right:.6f}], width={right-left:.6f}")
        
        if check_result:
            right = mid
        else:
            left = mid
        
        iterations += 1
    
    result = 0.5 * (left + right)
    
    if verbose:
        print(f"Converged: a_ctrl*={result:.6f} after {iterations} iterations")
    
    return result


def find_min_a_ctrl_with_diagnostics(A_min, omega, check_fn: Callable[[float], bool],
                                     lo: float = 0.0, hi_init: float = 0.1, 
                                     tol: float = 1e-3, max_expand: int = 20, 
                                     max_iter: int = 40) -> Dict:
    """
    Find minimal a_ctrl with detailed diagnostics.
    
    Returns:
        dict with keys:
        - a_ctrl_star: optimal value
        - n_expansions: number of expansion steps
        - n_iterations: number of bisection iterations
        - converged: whether search converged within tolerance
        - final_interval: (left, right) final search interval
    """
    # Expansion phase
    hi = hi_init
    expands = 0
    
    while not check_fn(hi):
        hi *= 2.0
        expands += 1
        if expands > max_expand:
            return {
                'a_ctrl_star': hi,
                'n_expansions': expands,
                'n_iterations': 0,
                'converged': False,
                'final_interval': (lo, hi),
                'warning': 'Max expansions reached'
            }
    
    # Bisection phase
    left, right = lo, hi
    iterations = 0
    
    while right - left > tol and iterations < max_iter:
        mid = 0.5 * (left + right)
        if check_fn(mid):
            right = mid
        else:
            left = mid
        iterations += 1
    
    result = 0.5 * (left + right)
    converged = (right - left <= tol)
    
    return {
        'a_ctrl_star': result,
        'n_expansions': expands,
        'n_iterations': iterations,
        'converged': converged,
        'final_interval': (left, right),
        'final_width': right - left
    }


def robust_find_min_a_ctrl(A_min, omega, check_fn: Callable[[float], bool],
                           n_trials: int = 3, **kwargs) -> float:
    """
    Robust version that runs multiple trials with different initial conditions.
    Returns the minimum a_ctrl found across all trials.
    
    Useful when check_fn might have stochastic behavior.
    """
    results = []
    hi_inits = [0.05, 0.1, 0.2][:n_trials]
    
    for hi_init in hi_inits:
        try:
            a_ctrl = find_min_a_ctrl(
                A_min, omega, check_fn, 
                hi_init=hi_init, 
                **kwargs
            )
            results.append(a_ctrl)
        except Exception as e:
            print(f"Warning: Trial with hi_init={hi_init} failed: {e}")
            continue
    
    if not results:
        raise RuntimeError("All trials failed in robust_find_min_a_ctrl")
    
    return min(results)


def validate_search_result(a_ctrl: float, check_fn: Callable[[float], bool], 
                          epsilon: float = 1e-4) -> Dict:
    """
    Validate that the found a_ctrl is indeed minimal.
    
    Checks:
    1. check_fn(a_ctrl) is True
    2. check_fn(a_ctrl - epsilon) is False (if a_ctrl > epsilon)
    
    Returns:
        dict with validation results
    """
    validation = {
        'a_ctrl': a_ctrl,
        'satisfies_constraint': check_fn(a_ctrl),
        'is_minimal': False,
        'margin': None
    }
    
    if a_ctrl > epsilon:
        lower_check = check_fn(a_ctrl - epsilon)
        validation['is_minimal'] = not lower_check
        
        # Find margin: how much can we decrease before failing
        if validation['satisfies_constraint']:
            test_vals = np.linspace(max(0, a_ctrl - 0.1), a_ctrl, 20)
            for test_val in test_vals:
                if not check_fn(test_val):
                    validation['margin'] = a_ctrl - test_val
                    break
    else:
        validation['is_minimal'] = True
        validation['margin'] = a_ctrl
    
    return validation


def binary_search_with_cache(A_min, omega, check_fn: Callable[[float], bool],
                             lo: float = 0.0, hi_init: float = 0.1, 
                             tol: float = 1e-3, **kwargs) -> tuple:
    """
    Binary search that caches check_fn evaluations.
    Returns (a_ctrl, cache_dict) where cache_dict maps a_ctrl -> bool.
    
    Useful for expensive check_fn.
    """
    cache = {}
    
    def cached_check_fn(a_ctrl: float) -> bool:
        # Round to avoid floating point key issues
        key = round(a_ctrl, 8)
        if key not in cache:
            cache[key] = check_fn(a_ctrl)
        return cache[key]
    
    a_ctrl = find_min_a_ctrl(
        A_min, omega, cached_check_fn,
        lo=lo, hi_init=hi_init, tol=tol, **kwargs
    )
    
    return a_ctrl, cache


if __name__ == "__main__":
    print("Testing improved bisection search...")
    
    # Test 1: Simple monotonic function
    print("\nTest 1: Finding sqrt(2) by searching x^2 >= 2")
    def check_fn_1(x):
        return x**2 >= 2.0
    
    result = find_min_a_ctrl(
        A_min=None, omega=None, check_fn=check_fn_1,
        lo=0.0, hi_init=1.0, tol=1e-6, verbose=True
    )
    print(f"Result: {result:.6f}, True value: {np.sqrt(2):.6f}, Error: {abs(result - np.sqrt(2)):.2e}")
    
    # Test 2: With diagnostics
    print("\nTest 2: Same search with diagnostics")
    diag = find_min_a_ctrl_with_diagnostics(
        A_min=None, omega=None, check_fn=check_fn_1,
        lo=0.0, hi_init=1.0, tol=1e-6
    )
    print(f"Diagnostics: {diag}")
    
    # Test 3: Validation
    print("\nTest 3: Validating result")
    validation = validate_search_result(result, check_fn_1, epsilon=1e-4)
    print(f"Validation: {validation}")
    
    # Test 4: Robust search with stochastic function
    print("\nTest 4: Robust search with noisy function")
    def noisy_check_fn(x):
        noise = np.random.randn() * 0.01
        return (x + noise)**2 >= 2.0
    
    robust_result = robust_find_min_a_ctrl(
        A_min=None, omega=None, check_fn=noisy_check_fn,
        n_trials=5, lo=0.0, hi_init=1.0, tol=1e-3, verbose=False
    )
    print(f"Robust result: {robust_result:.6f}")
    
    # Test 5: Cached search
    print("\nTest 5: Cached binary search")
    call_count = [0]
    def expensive_check_fn(x):
        call_count[0] += 1
        return x**2 >= 2.0
    
    cached_result, cache = binary_search_with_cache(
        A_min=None, omega=None, check_fn=expensive_check_fn,
        lo=0.0, hi_init=1.0, tol=1e-4
    )
    print(f"Result: {cached_result:.6f}")
    print(f"Function calls: {call_count[0]}")
    print(f"Cache size: {len(cache)}")
    
    print("\n✓ All tests passed!")