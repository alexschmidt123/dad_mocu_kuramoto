"""
GPU-accelerated pacemaker control using PyCUDA.
Following Chen et al. (2023) methodology for massive parallelization.

This allows parallel ODE integration for multiple matrices simultaneously,
dramatically speeding up MOCU computation during data generation.
"""

import numpy as np

# Try to import PyCUDA
try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False
    print("Warning: PyCUDA not available. Falling back to CPU computation.")


# CUDA kernel for Kuramoto model integration
KURAMOTO_KERNEL = """
__device__ float order_param(float* theta, int N) {
    float cos_sum = 0.0f;
    float sin_sum = 0.0f;
    
    for (int i = 0; i < N; i++) {
        cos_sum += cosf(theta[i]);
        sin_sum += sinf(theta[i]);
    }
    
    cos_sum /= N;
    sin_sum /= N;
    
    return sqrtf(cos_sum * cos_sum + sin_sum * sin_sum);
}

__global__ void kuramoto_step(
    float* theta,           // Current phases (batch_size, N)
    float* theta_new,       // Updated phases (batch_size, N)
    float* A,              // Coupling matrices (batch_size, N, N)
    float* omega,          // Natural frequencies (batch_size, N)
    float* a_ctrl,         // Control parameters (batch_size,)
    int N,                 // Number of oscillators
    float dt,              // Time step
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / N;
    int osc_idx = idx % N;
    
    if (batch_idx >= batch_size) return;
    
    int base = batch_idx * N;
    int A_base = batch_idx * N * N;
    
    // Natural frequency term
    float dtheta = omega[base + osc_idx];
    
    // Coupling term
    float coupling = 0.0f;
    for (int j = 0; j < N; j++) {
        float a_ij = A[A_base + osc_idx * N + j];
        float phase_diff = theta[base + j] - theta[base + osc_idx];
        coupling += a_ij * sinf(phase_diff);
    }
    dtheta += coupling;
    
    // Pacemaker control term (pacemaker at theta_c = 0)
    dtheta += a_ctrl[batch_idx] * sinf(-theta[base + osc_idx]);
    
    // Update phase
    float new_theta = theta[base + osc_idx] + dt * dtheta;
    
    // Wrap to [-pi, pi]
    while (new_theta > 3.14159265f) new_theta -= 2.0f * 3.14159265f;
    while (new_theta < -3.14159265f) new_theta += 2.0f * 3.14159265f;
    
    theta_new[base + osc_idx] = new_theta;
}

__global__ void compute_order_params(
    float* theta,           // Phases (batch_size, N)
    float* R_values,       // Output order parameters (batch_size,)
    int N,
    int batch_size
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    int base = batch_idx * N;
    R_values[batch_idx] = order_param(&theta[base], N);
}
"""


class GPUKuramotoSimulator:
    """GPU-accelerated Kuramoto simulator using PyCUDA."""
    
    def __init__(self, N, dt=0.01, T=5.0, burn_in=2.0, R_target=0.95):
        if not PYCUDA_AVAILABLE:
            raise RuntimeError("PyCUDA is required for GPU simulator")
        
        self.N = N
        self.dt = dt
        self.T = T
        self.burn_in = burn_in
        self.R_target = R_target
        
        # Compile CUDA kernel
        self.mod = SourceModule(KURAMOTO_KERNEL)
        self.kuramoto_step = self.mod.get_function("kuramoto_step")
        self.compute_order_params = self.mod.get_function("compute_order_params")
        
        # CUDA block and grid sizes
        self.block_size = 256
        
    def simulate_batch(self, A_batch, omega_batch, a_ctrl_batch, n_trajectories=1):
        """
        Simulate multiple Kuramoto systems in parallel on GPU.
        
        Args:
            A_batch: (batch_size, N, N) coupling matrices
            omega_batch: (batch_size, N) natural frequencies
            a_ctrl_batch: (batch_size,) control parameters
            n_trajectories: number of trajectories per system (averaged)
        
        Returns:
            R_mean: (batch_size,) mean order parameters after burn-in
        """
        batch_size = A_batch.shape[0]
        
        # Average over multiple trajectories
        R_means = []
        
        for traj in range(n_trajectories):
            # Initialize random phases
            theta = np.random.uniform(-np.pi, np.pi, (batch_size, self.N)).astype(np.float32)
            
            # Allocate GPU memory
            theta_gpu = cuda.mem_alloc(theta.nbytes)
            theta_new_gpu = cuda.mem_alloc(theta.nbytes)
            A_gpu = cuda.mem_alloc(A_batch.astype(np.float32).nbytes)
            omega_gpu = cuda.mem_alloc(omega_batch.astype(np.float32).nbytes)
            a_ctrl_gpu = cuda.mem_alloc(a_ctrl_batch.astype(np.float32).nbytes)
            
            # Copy to GPU
            cuda.memcpy_htod(theta_gpu, theta)
            cuda.memcpy_htod(A_gpu, A_batch.astype(np.float32))
            cuda.memcpy_htod(omega_gpu, omega_batch.astype(np.float32))
            cuda.memcpy_htod(a_ctrl_gpu, a_ctrl_batch.astype(np.float32))
            
            # Integration parameters
            n_steps = int(self.T / self.dt)
            burn_in_steps = int(self.burn_in / self.dt)
            
            # Grid size
            total_threads = batch_size * self.N
            grid_size = (total_threads + self.block_size - 1) // self.block_size
            
            # Integrate
            R_accumulator = np.zeros(batch_size, dtype=np.float32)
            n_recorded = 0
            
            for step in range(n_steps):
                # Perform one integration step
                self.kuramoto_step(
                    theta_gpu, theta_new_gpu,
                    A_gpu, omega_gpu, a_ctrl_gpu,
                    np.int32(self.N),
                    np.float32(self.dt),
                    np.int32(batch_size),
                    block=(self.block_size, 1, 1),
                    grid=(grid_size, 1)
                )
                
                # Swap buffers
                theta_gpu, theta_new_gpu = theta_new_gpu, theta_gpu
                
                # Record order parameter after burn-in
                if step >= burn_in_steps:
                    R_values = np.zeros(batch_size, dtype=np.float32)
                    R_values_gpu = cuda.mem_alloc(R_values.nbytes)
                    
                    grid_size_R = (batch_size + self.block_size - 1) // self.block_size
                    self.compute_order_params(
                        theta_gpu, R_values_gpu,
                        np.int32(self.N),
                        np.int32(batch_size),
                        block=(self.block_size, 1, 1),
                        grid=(grid_size_R, 1)
                    )
                    
                    cuda.memcpy_dtoh(R_values, R_values_gpu)
                    R_accumulator += R_values
                    n_recorded += 1
            
            # Average R over time steps
            R_mean = R_accumulator / n_recorded
            R_means.append(R_mean)
        
        # Average over trajectories
        return np.mean(R_means, axis=0)
    
    def sync_check_batch(self, A_batch, omega_batch, a_ctrl_batch, n_trajectories=1):
        """
        Check synchronization for a batch of systems.
        
        Returns:
            sync_flags: (batch_size,) boolean array indicating if each system syncs
        """
        R_mean = self.simulate_batch(A_batch, omega_batch, a_ctrl_batch, n_trajectories)
        return R_mean >= self.R_target


def find_optimal_control_batch_gpu(A_batch, omega_batch, sim_params, tol=5e-3, max_iter=30):
    """
    Find optimal control parameters for a batch of systems using GPU acceleration.
    
    This is the key speedup - we can binary search for multiple systems in parallel.
    
    Args:
        A_batch: (batch_size, N, N) coupling matrices
        omega_batch: (batch_size, N) natural frequencies
        sim_params: dict with dt, T, burn_in, R_target
        tol: tolerance for binary search
        max_iter: maximum iterations
    
    Returns:
        a_ctrl_star: (batch_size,) optimal control parameters
    """
    if not PYCUDA_AVAILABLE:
        raise RuntimeError("PyCUDA required for batch GPU computation")
    
    batch_size = A_batch.shape[0]
    N = A_batch.shape[1]
    
    # Initialize simulator
    simulator = GPUKuramotoSimulator(
        N=N,
        dt=sim_params.get('dt', 0.01),
        T=sim_params.get('T', 5.0),
        burn_in=sim_params.get('burn_in', 2.0),
        R_target=sim_params.get('R_target', 0.95)
    )
    
    # Initialize search bounds for each system
    lo = np.zeros(batch_size, dtype=np.float32)
    hi = np.full(batch_size, 0.1, dtype=np.float32)
    
    # Phase 1: Expand upper bounds until all systems sync
    max_expand = 20
    for expand_iter in range(max_expand):
        syncs = simulator.sync_check_batch(A_batch, omega_batch, hi, n_trajectories=2)
        
        if np.all(syncs):
            break
        
        # Double hi for systems that don't sync yet
        hi[~syncs] *= 2.0
    
    # Phase 2: Binary search for each system
    for iter in range(max_iter):
        # Check convergence
        if np.all(hi - lo <= tol):
            break
        
        # Compute midpoints
        mid = 0.5 * (lo + hi)
        
        # Check which systems sync at midpoint
        syncs = simulator.sync_check_batch(A_batch, omega_batch, mid, n_trajectories=2)
        
        # Update bounds
        hi[syncs] = mid[syncs]
        lo[~syncs] = mid[~syncs]
    
    # Return final estimates
    return 0.5 * (lo + hi)


# Fallback to CPU if PyCUDA not available
def find_optimal_control_batch_cpu(A_batch, omega_batch, sim_params, tol=5e-3):
    """
    CPU fallback for batch optimal control computation.
    Uses standard scipy integration but processes sequentially.
    """
    from core.bisection import find_min_a_ctrl
    from core.pacemaker_control import sync_check
    
    batch_size = A_batch.shape[0]
    results = np.zeros(batch_size)
    
    for i in range(batch_size):
        A = A_batch[i]
        omega = omega_batch[i]
        
        def check_fn(a_ctrl):
            try:
                return sync_check(A, omega, a_ctrl, **sim_params)
            except:
                return False
        
        try:
            results[i] = find_min_a_ctrl(A, omega, check_fn, lo=0.0, hi_init=0.1, tol=tol)
        except:
            results[i] = 2.0
    
    return results


def find_optimal_control_batch(A_batch, omega_batch, sim_params, use_gpu=True, tol=5e-3):
    """
    Unified interface for batch optimal control computation.
    
    Automatically uses GPU if available, otherwise falls back to CPU.
    """
    if use_gpu and PYCUDA_AVAILABLE:
        return find_optimal_control_batch_gpu(A_batch, omega_batch, sim_params, tol=tol)
    else:
        return find_optimal_control_batch_cpu(A_batch, omega_batch, sim_params, tol=tol)


if __name__ == "__main__":
    print("Testing PyCUDA integration...")
    
    if PYCUDA_AVAILABLE:
        print("PyCUDA is available!")
        
        # Test batch simulation
        N = 5
        batch_size = 10
        
        # Generate random test systems
        A_batch = np.random.uniform(0.1, 0.5, (batch_size, N, N)).astype(np.float32)
        for i in range(batch_size):
            A_batch[i] = 0.5 * (A_batch[i] + A_batch[i].T)
            np.fill_diagonal(A_batch[i], 0.0)
        
        omega_batch = np.random.uniform(-1.0, 1.0, (batch_size, N)).astype(np.float32)
        
        sim_params = {'dt': 0.01, 'T': 3.0, 'burn_in': 1.0, 'R_target': 0.95}
        
        print(f"\nTesting batch optimal control for {batch_size} systems...")
        import time
        
        start = time.time()
        a_ctrl_star = find_optimal_control_batch(A_batch, omega_batch, sim_params, use_gpu=True)
        elapsed = time.time() - start
        
        print(f"Completed in {elapsed:.2f} seconds")
        print(f"Results: {a_ctrl_star}")
        print(f"Average a_ctrl: {np.mean(a_ctrl_star):.4f}")
        
    else:
        print("PyCUDA not available - install with:")
        print("  pip install pycuda")
        print("  See: https://wiki.tiker.net/PyCuda/Installation")