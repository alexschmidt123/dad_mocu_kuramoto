import numpy as np

def euler_integrate(theta0, vecfield, dt, steps):
    theta = theta0.copy(); traj = [theta.copy()]
    for _ in range(steps):
        theta = theta + dt * vecfield(theta)
        theta = (theta + np.pi) % (2*np.pi) - np.pi
        traj.append(theta.copy())
    return np.array(traj)

def order_param(theta):
    z = np.exp(1j*theta)
    return np.abs(np.mean(z, axis=-1))

def simulate_with_pacemaker(A, omega, a_ctrl, dt=0.01, T=5.0, burn_in=2.0):
    N = len(omega)
    theta0 = np.random.uniform(-np.pi, np.pi, size=N)
    def vf(theta):
        pacer = a_ctrl * np.sin(-theta)  # θ_c = 0
        coupling = np.zeros_like(theta)
        for i in range(N):
            coupling[i] = np.sum(A[i] * np.sin(theta - theta[i]))
        return omega + coupling + pacer
    steps = int(T/dt)
    traj = euler_integrate(theta0, vf, dt, steps)
    R = order_param(traj)
    t = np.linspace(0.0, T, steps+1)
    mask = t >= burn_in
    R_mean = float(np.mean(R[mask])) if np.any(mask) else float(np.mean(R))
    return dict(t=t, theta=traj, R=R, R_mean=R_mean)

def sync_check(A, omega, a_ctrl, R_target=0.95, dt=0.01, T=5.0, burn_in=2.0):
    out = simulate_with_pacemaker(A, omega, a_ctrl, dt=dt, T=T, burn_in=burn_in)
    return out["R_mean"] >= R_target
