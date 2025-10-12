# Sequential Experiment Design for Kuramoto Oscillator Networks

A machine learning-accelerated framework for optimal experimental design on Kuramoto oscillator networks with pacemaker control, implementing Deep Adaptive Design (DAD) and Mean Objective Cost of Uncertainty (MOCU) optimization.

## Overview

This project implements sequential experiment design for learning the coupling structure of Kuramoto oscillator networks through active pair-wise synchronization tests. The goal is to efficiently determine the minimal pacemaker coupling needed to synchronize the network under uncertainty.

### Key Features

- **MOCU-based Optimization**: Quantifies the impact of uncertainty on control objectives
- **MPNN Surrogate Model**: Fast approximation of expensive computations using graph neural networks
- **Multiple Design Strategies**: 
  - Random baseline
  - Greedy Expected Remaining MOCU (ERM)
  - Deep Adaptive Design (DAD) via behavior cloning
- **GPU Acceleration**: CUDA-optimized training with automatic mixed precision
- **Robust ODE Integration**: Scipy-based RK45 with trajectory averaging

### Scientific Background

The Kuramoto model describes synchronization in coupled oscillator systems:

$$\dot{\theta}_i = \omega_i + \sum_{j} a_{ij}\sin(\theta_j - \theta_i) + a_{\text{ctrl}}\sin(\theta_c - \theta_i)$$

where:
- $\theta_i$: phase of oscillator $i$
- $\omega_i$: natural frequency
- $a_{ij}$: coupling strength (unknown)
- $a_{\text{ctrl}}$: pacemaker control parameter (to be minimized)

## Installation

### Prerequisites

- Python 3.10
- CUDA 11.8 or 12.1 (for GPU acceleration)
- Conda (recommended) or pip

### Quick Setup with Conda

```bash
# Clone repository
git clone <your-repo-url>
cd dad_mocu_kuramoto

# Create and activate environment
conda env create -f environment.yml
conda activate dad_mocu_kuramoto

# Verify installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Alternative: Setup with pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Run Tests

Verify everything works correctly:

```bash
python comprehensive_test.py
```

Expected output: `9/9 tests passed`

### 2. Quick Demo

Run a quick demonstration with single episodes:

```bash
python main_demo.py --mode quick
```

### 3. Full Evaluation

Run comprehensive evaluation with multiple episodes:

```bash
python main_demo.py --mode full --episodes 20
```

### 4. Custom Configuration

Edit `configs/exp_fixedK.yaml` to customize parameters, then run:

```bash
python main_demo.py --mode full --episodes 10 --save-results my_results.json
```

## Project Structure

```
dad_mocu_kuramoto/
├── configs/
│   └── exp_fixedK.yaml          # Configuration parameters
├── core/                         # Core simulation components
│   ├── belief.py                # Belief state management
│   ├── kuramoto_env.py          # Experiment environment
│   ├── pacemaker_control.py     # ODE integration & synchronization
│   └── bisection.py             # Binary search for optimal control
├── data_generation/
│   └── synthetic_data.py        # Training data generation with proper MOCU/ERM
├── surrogate/                   # Neural network surrogate
│   ├── mpnn_surrogate.py        # Graph neural network architecture
│   └── train_surrogate.py       # CUDA-optimized training pipeline
├── design/                      # Experiment selection strategies
│   ├── greedy_erm.py           # Greedy ERM baseline
│   ├── dad_policy.py           # DAD policy network
│   └── train_bc.py             # Behavior cloning training
├── eval/                        # Evaluation and metrics
│   ├── run_eval.py             # Episode execution
│   └── metrics.py              # Statistics and visualization
├── main_demo.py                 # Main entry point
├── comprehensive_test.py        # Test suite
└── environment.yml              # Conda environment specification
```

## Configuration

Key parameters in `configs/exp_fixedK.yaml`:

### System Parameters
```yaml
N: 5                    # Number of oscillators
K: 4                    # Number of experiments (fixed budget)
prior_lower: 0.05       # Lower bound of coupling prior
prior_upper: 0.50       # Upper bound of coupling prior
```

### Simulation Settings
```yaml
sim:
  dt: 0.01              # Time step
  T: 5.0                # Simulation time
  burn_in: 2.0          # Transient time to discard
  R_target: 0.95        # Synchronization threshold
  method: "RK45"        # Integration method (RK45 or Euler)
  n_trajectories: 3     # Trajectories to average
```

### Training Parameters
```yaml
surrogate:
  n_train: 200          # Training samples (demo mode)
  n_theta_samples: 10   # Monte Carlo samples for MOCU/ERM
  epochs: 30            # Training epochs
  batch_size: 128       # Batch size (adjust for GPU memory)
  hidden: 64            # Hidden dimension
```

## Usage Examples

### Basic Usage

```python
from core.kuramoto_env import PairTestEnv
from surrogate.mpnn_surrogate import MPNNSurrogate
from design.greedy_erm import choose_next_pair_greedy

# Create environment
N, K = 5, 4
omega = np.random.uniform(-1.0, 1.0, N)
surrogate = MPNNSurrogate(mocu_scale=1.0)
env = PairTestEnv(N=N, omega=omega, prior_bounds=(0.05, 0.50), K=K, surrogate=surrogate)

# Run one experiment
candidates = env.candidate_pairs()
xi = choose_next_pair_greedy(env, candidates)
result = env.step(xi)
print(f"Tested pair {xi}, outcome: {result['y']}")
```

### Custom Strategy

```python
def my_strategy(env, candidates):
    # Your custom pair selection logic
    return candidates[0]  # Example: always pick first pair

# Use it in evaluation
from eval.run_eval import run_episode
result = run_episode(env, my_strategy, sim_opts)
```

## Performance

### Demo Mode (Default)
- **Training samples**: 200
- **MOCU estimation**: 10 MC samples
- **Runtime**: ~15-20 minutes (CPU), ~12-15 minutes (GPU)

### Production Mode
Edit config for better results:
```yaml
surrogate:
  n_train: 2000
  n_theta_samples: 30
  epochs: 100
```
- **Runtime**: ~1.5-2 hours (CPU), ~45-60 minutes (GPU)

### GPU Acceleration
- **Training speedup**: 2-5x faster
- **Overall speedup**: ~1.3-2x (ODE integration is CPU-bound)
- **Recommended**: 8+ GB GPU for batch_size=128

## Output

The demo generates:

1. **Terminal Output**: Real-time progress and statistics
2. **Plots**:
   - `mocu_curves.png`: MOCU evolution over experiment steps
   - `a_ctrl_distribution.png`: Distribution of final control parameters
3. **Results File** (optional): JSON with detailed metrics

### Example Results

```
Strategy Comparison:
  Random:       a_ctrl* = 0.152 ± 0.031, MOCU = 0.046 ± 0.012
  GreedyERM:    a_ctrl* = 0.125 ± 0.023, MOCU = 0.032 ± 0.009 (↓18% / ↓30%)
  DAD:          a_ctrl* = 0.120 ± 0.020, MOCU = 0.029 ± 0.008 (↓21% / ↓37%)
```

## Testing

Run the comprehensive test suite:

```bash
python comprehensive_test.py
```

Tests cover:
- Module imports
- Belief system operations
- Pacemaker control simulation
- Bisection search
- Surrogate model
- Environment and strategies
- Evaluation pipeline
- DAD policy

## Troubleshooting

### CUDA Out of Memory
```yaml
# Reduce batch size in configs/exp_fixedK.yaml
surrogate:
  batch_size: 32  # or 16
```

### Slow Training
- Check GPU utilization: `nvidia-smi`
- Increase batch size if GPU underutilized
- Note: Data generation is CPU-bound (ODE integration)

### Test Failures
```bash
# Ensure scipy installed
conda install scipy

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

## References

### Related Work
- **Kuramoto Model**: Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence
- **MOCU Framework**: IEEE paper on accelerating OED for Kuramoto models (2021)
- **Deep Adaptive Design**: Foster et al., Deep Adaptive Design

### Repositories
- [Kuramoto Model OED Acceleration (2021)](https://github.com/bjyoontamu/Kuramoto-Model-OED-acceleration)
- [MPNN OED Acceleration (2023)](https://github.com/Levishery/AccelerateOED)
- [Deep Adaptive Design](https://github.com/ae-foster/dad)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{kuramoto_oed_2025,
  title={Sequential Experiment Design for Kuramoto Oscillator Networks},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo}
}
```

## License

This project is for educational and research purposes. See LICENSE file for details.

## Contact

For questions or issues:
- Open an issue on GitHub
- Contact: your.email@example.com

---

**Status**: Production-ready with comprehensive testing
**Last Updated**: 2025