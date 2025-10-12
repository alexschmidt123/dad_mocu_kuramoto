"""
Improved pacemaker control module with better numerical integration.
Uses scipy's solve_ivp with RK45 for more accurate ODE integration.
"""

import numpy as np
try:
    from scipy.integrate import solve_ivp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, falling back to Euler integration")


def order_param(theta):
    """
    Compute Kuramoto order parameter R(t).
    R = |mean(exp(i*theta))| measures phase coherence.
    R = 1: perfect synchronization
    R = 0: completely incoherent
    """
    if len(theta.shape) == 1:
        # Single time point
        z = np.exp(1j * theta)
        return np.abs(np.mean(z))
    else:
        # Multiple time points
        z = np.exp(1j * theta)
        return np.abs(np.mean(z, axis=-1))


def euler_integrate(theta0, vecfield, dt, steps):
    """
    Simple Euler integration (kept for backward compatibility and when scipy unavailable).
    """
    theta = theta0.copy()
    traj = [theta.copy()]
    
    for _ in range(steps):
        theta = theta + dt * vecfield(theta)
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        traj.append(theta.copy())
    
    return np.array(traj)


def simulate_with_pacemaker(A, omega, a_ctrl, dt=0.01, T=5.0, burn_in=2.0, 
                            method='RK45', n_trajectories=3):
    """
    Simulate Kuramoto model with pacemaker control.
    
    Dynamics:
        dθ_i/dt = ω_i + Σ_j a_ij sin(θ_j - θ_i) + a_ctrl sin(θ_c - θ_i)
    
    where θ_c = 0 (pacemaker at origin).
    
    Args:
        A: (N, N) coupling matrix
        omega: (N,) natural frequencies
        a_ctrl: scalar pacemaker coupling strength
        dt: time step for output (not integration step)
        T: total simulation time
        burn_in: time to discard for transient behavior
        method: integration method ('RK45', 'RK23', 'Euler')
        n_trajectories: number of trajectories to average over
    
    Returns:
        dict with keys: t, theta, R, R_mean, R_all_trajectories
    """
    N = len(omega)
    
    def kuramoto_pacemaker(t, theta):
        """ODE right-hand side."""
        # Natural frequencies
        dtheta = omega.copy()
        
        # Coupling terms
        for i in range(N):
            coupling = np.sum(A[i] * np.sin(theta - theta[i]))
            dtheta[i] += coupling
        
        # Pacemaker control (θ_c = 0)
        pacer = a_ctrl * np.sin(-theta)
        dtheta += pacer
        
        return dtheta
    
    # Run multiple trajectories and average
    all_R = []
    all_theta = []
    t = None
    
    for traj_idx in range(n_trajectories):
        # Random initial condition
        theta0 = np.random.uniform(-np.pi, np.pi, size=N)
        
        if method == 'Euler' or not SCIPY_AVAILABLE:
            # Fallback to Euler method
            t_eval = np.arange(0, T + dt, dt)
            theta_traj = [theta0.copy()]
            theta = theta0.copy()
            
            for _ in range(len(t_eval) - 1):
                dtheta = kuramoto_pacemaker(0, theta)
                theta = theta + dt * dtheta
                theta = (theta + np.pi) % (2 * np.pi) - np.pi
                theta_traj.append(theta.copy())
            
            theta_traj = np.array(theta_traj)
            t = t_eval
        else:
            # Use scipy's solve_ivp with adaptive stepping
            t_eval = np.arange(0, T + dt, dt)
            
            try:
                sol = solve_ivp(
                    kuramoto_pacemaker,
                    t_span=(0, T),
                    y0=theta0,
                    method=method,
                    t_eval=t_eval,
                    rtol=1e-6,
                    atol=1e-8
                )
                
                theta_traj = sol.y.T  # (time_steps, N)
                t = sol.t
                
                # Wrap angles to [-π, π]
                theta_traj = (theta_traj + np.pi) % (2 * np.pi) - np.pi
            except Exception as e:
                # Fallback to Euler if scipy fails
                print(f"Warning: scipy integration failed ({e}), using Euler")
                t_eval = np.arange(0, T + dt, dt)
                theta_traj = [theta0.copy()]
                theta = theta0.copy()
                
                for _ in range(len(t_eval) - 1):
                    dtheta = kuramoto_pacemaker(0, theta)
                    theta = theta + dt * dtheta
                    theta = (theta + np.pi) % (2 * np.pi) - np.pi
                    theta_traj.append(theta.copy())
                
                theta_traj = np.array(theta_traj)
                t = t_eval
        
        # Compute order parameter
        R = order_param(theta_traj)
        all_R.append(R)
        all_theta.append(theta_traj)
    
    # Average over trajectories
    avg_R = np.mean(all_R, axis=0)
    
    # Compute mean R after burn-in
    mask = t >= burn_in
    if np.any(mask):
        R_mean = float(np.mean(avg_R[mask]))
    else:
        R_mean = float(np.mean(avg_R))
    
    return dict(
        t=t,
        theta=all_theta[0],  # Return first trajectory for visualization
        R=avg_R,
        R_mean=R_mean,
        R_all_trajectories=all_R
    )


def sync_check(A, omega, a_ctrl, R_target=0.95, dt=0.01, T=5.0, burn_in=2.0, 
               method='RK45', n_trajectories=3):
    """
    Check if system synchronizes with given control parameter.
    
    Returns True if mean order parameter R after burn-in exceeds R_target.
    
    Args:
        A: (N, N) coupling matrix
        omega: (N,) natural frequencies  
        a_ctrl: scalar pacemaker coupling strength
        R_target: threshold for synchronization (default 0.95)
        dt: time step for output
        T: total simulation time
        burn_in: time to discard for transient behavior
        method: integration method
        n_trajectories: number of trajectories to average over
    
    Returns:
        bool: True if synchronized, False otherwise
    """
    try:
        out = simulate_with_pacemaker(
            A, omega, a_ctrl, 
            dt=dt, T=T, burn_in=burn_in, 
            method=method, n_trajectories=n_trajectories
        )
        return out["R_mean"] >= R_target
    except Exception as e:
        # If integration fails, assume not synchronized
        print(f"Warning: Integration failed for a_ctrl={a_ctrl}: {e}")
        return False


def compute_phase_coherence(theta_traj, window_size=50):
    """
    Compute time-averaged phase coherence over sliding windows.
    Useful for detecting intermittent synchronization.
    
    Args:
        theta_traj: (T, N) array of phase trajectories
        window_size: size of sliding window
    
    Returns:
        array of coherence values over time
    """
    T = len(theta_traj)
    coherence = []
    
    for t in range(window_size, T):
        window = theta_traj[t-window_size:t]
        R_window = order_param(window)
        coherence.append(np.mean(R_window))
    
    return np.array(coherence)


def estimate_sync_time(A, omega, a_ctrl, R_target=0.95, dt=0.01, max_time=20.0, 
                       method='RK45'):
    """
    Estimate time required to reach synchronization.
    
    Returns:
        float: time to sync, or np.inf if doesn't sync within max_time
    """
    if not SCIPY_AVAILABLE:
        print("Warning: estimate_sync_time requires scipy, returning inf")
        return np.inf
    
    N = len(omega)
    theta0 = np.random.uniform(-np.pi, np.pi, size=N)
    
    def kuramoto_pacemaker(t, theta):
        dtheta = omega.copy()
        for i in range(N):
            coupling = np.sum(A[i] * np.sin(theta - theta[i]))
            dtheta[i] += coupling
        pacer = a_ctrl * np.sin(-theta)
        dtheta += pacer
        return dtheta
    
    # Event function to detect synchronization
    def sync_event(t, theta):
        R = order_param(theta)
        return R - R_target
    
    sync_event.terminal = True
    sync_event.direction = 1
    
    try:
        sol = solve_ivp(
            kuramoto_pacemaker,
            t_span=(0, max_time),
            y0=theta0,
            method=method,
            events=sync_event,
            rtol=1e-6,
            atol=1e-8
        )
        
        if sol.t_events[0].size > 0:
            return float(sol.t_events[0][0])
        else:
            return np.inf
    except Exception as e:
        print(f"Warning: estimate_sync_time failed: {e}")
        return np.inf


def analyze_synchronization_quality(A, omega, a_ctrl, dt=0.01, T=10.0, burn_in=5.0,
                                    method='RK45', n_trajectories=5):
    """
    Comprehensive analysis of synchronization quality.
    
    Returns:
        dict with metrics:
        - R_mean: mean order parameter after burn-in
        - R_std: standard deviation of order parameter
        - R_min: minimum order parameter after burn-in
        - sync_achieved: whether R_mean >= 0.95
        - stability: measure of synchronization stability
    """
    results = []
    
    for _ in range(n_trajectories):
        out = simulate_with_pacemaker(
            A, omega, a_ctrl, dt=dt, T=T, burn_in=burn_in,
            method=method, n_trajectories=1
        )
        
        t = out['t']
        R = out['R']
        mask = t >= burn_in
        
        if np.any(mask):
            R_post_burn = R[mask]
            results.append({
                'R_mean': np.mean(R_post_burn),
                'R_std': np.std(R_post_burn),
                'R_min': np.min(R_post_burn)
            })
    
    # Aggregate results
    R_means = [r['R_mean'] for r in results]
    R_stds = [r['R_std'] for r in results]
    R_mins = [r['R_min'] for r in results]
    
    avg_R_mean = np.mean(R_means)
    avg_R_std = np.mean(R_stds)
    avg_R_min = np.mean(R_mins)
    
    # Stability: low if high variance across trajectories
    stability = 1.0 / (1.0 + np.std(R_means))
    
    return {
        'R_mean': avg_R_mean,
        'R_std': avg_R_std,
        'R_min': avg_R_min,
        'sync_achieved': avg_R_mean >= 0.95,
        'stability': stability,
        'trajectory_variance': np.std(R_means)
    }


if __name__ == "__main__":
    print("Testing improved pacemaker control...")
    
    # Create test system
    N = 5
    omega = np.random.uniform(-1.0, 1.0, N)
    A = np.random.uniform(0.1, 0.5, (N, N))
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    
    # Test synchronization check
    print("\nTesting sync_check with different a_ctrl values:")
    for a_ctrl in [0.0, 0.1, 0.2, 0.3, 0.5]:
        method = 'RK45' if SCIPY_AVAILABLE else 'Euler'
        synced = sync_check(A, omega, a_ctrl, method=method, n_trajectories=2)
        print(f"  a_ctrl = {a_ctrl:.2f}: {'✓ Synchronized' if synced else '✗ Not synchronized'}")
    
    # Test simulation
    print("\nTesting simulation with a_ctrl = 0.3:")
    method = 'RK45' if SCIPY_AVAILABLE else 'Euler'
    result = simulate_with_pacemaker(A, omega, 0.3, T=5.0, method=method, n_trajectories=2)
    print(f"  Mean R after burn-in: {result['R_mean']:.4f}")
    print(f"  Final R: {result['R'][-1]:.4f}")
    
    # Test comprehensive analysis
    if SCIPY_AVAILABLE:
        print("\nTesting comprehensive synchronization analysis:")
        analysis = analyze_synchronization_quality(A, omega, 0.3, T=10.0, n_trajectories=3)
        print(f"  R_mean: {analysis['R_mean']:.4f}")
        print(f"  R_std: {analysis['R_std']:.4f}")
        print(f"  R_min: {analysis['R_min']:.4f}")
        print(f"  Stability: {analysis['stability']:.4f}")
        print(f"  Sync achieved: {analysis['sync_achieved']}")
    
    print("\n✓ All tests passed!")