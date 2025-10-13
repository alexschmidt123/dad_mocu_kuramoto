# Deep Adaptive Design for Kuramoto Oscillator Networks

GPU-accelerated sequential experiment design for learning coupling structures in Kuramoto oscillator networks through active synchronization tests.

## Scientific Innovation

### Problem Formulation
Given a network of coupled Kuramoto oscillators:
$$\dot{\theta}_i = \omega_i + \sum_{j} a_{ij}\sin(\theta_j - \theta_i) + a_{\text{ctrl}}\sin(\theta_c - \theta_i)$$

**Goal**: Minimize the pacemaker control parameter $a_{\text{ctrl}}^*$ needed to synchronize the network, under uncertainty about coupling strengths $a_{ij}$.

### Mean Objective Cost of Uncertainty (MOCU)
We optimize experiment selection using MOCU:
$$\text{MOCU}(h) = \mathbb{E}_{\theta|h}\left[C(a_{\text{IBR}}(h), \theta) - C(a^*(\theta), \theta)\right]$$

where:
- $h$ represents current belief state
- $a_{\text{IBR}}(h)$ is information-based robust control
- $a^*(\theta)$ is optimal control for true parameters
- $C(a, \theta)$ is control cost (here, $a$ itself if system synchronizes)

### Key Contributions
1. **MPNN Surrogate**: Graph neural network approximates expensive MOCU/ERM computations (1000× speedup)
2. **Deep Adaptive Design**: Policy network trained via behavior cloning learns to select informative experiments
3. **GPU Optimization**: CUDA kernels + mixed precision training for RTX 4090

## Baseline Comparisons

| Strategy | Description | Reference |
|----------|-------------|-----------|
| **Random** | Uniformly sample candidate pairs | Standard baseline |
| **Fixed Design** | Pre-determined experiment sequence | Classical OED |
| **Greedy MPNN** | Minimize predicted ERM at each step | [Sherry et al. 2023](https://github.com/Levishery/AccelerateOED) |
| **DAD (Ours)** | Deep policy trained on greedy trajectories | [Foster et al. 2021](https://github.com/ae-foster/dad) |

## Repository Structure

```
dad_mocu_kuramoto/
├── main.py                      # Main training/evaluation script
├── configs/
│   └── config.yaml             # RTX 4090 optimized settings
├── core/                       # Simulation engine
│   ├── belief.py              # Bayesian belief updates
│   ├── kuramoto_env.py        # Experiment environment
│   ├── pacemaker_control.py   # ODE integration (RK45)
│   └── bisection.py           # Binary search for optimal control
├── data_generation/
│   └── synthetic_data.py      # Training data with proper MOCU/ERM
├── surrogate/                 # Neural network surrogate
│   ├── mpnn_surrogate.py      # Graph neural network
│   └── train_surrogate.py     # GPU-accelerated training
├── design/                    # Experiment selection strategies
│   ├── greedy_erm.py         # Greedy baseline
│   ├── dad_policy.py         # Deep adaptive design
│   └── train_bc.py           # Behavior cloning
└── eval/                      # Evaluation utilities
    ├── run_eval.py
    └── metrics.py
```

## Installation & Reproduction

### Prerequisites
- RTX 4090 GPU
- CUDA 12.1+
- Python 3.9+

### Setup
```bash
# Clone repository
git clone <repo-url>
cd dad_mocu_kuramoto

# Install dependencies (RTX 4090 optimized)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Run main pipeline
python main.py --mode full --episodes 50
```

### Expected Runtime
- Data generation: ~2 min (2000 samples, 30 MC samples)
- Surrogate training: ~3 min (100 epochs, batch=512, AMP)
- DAD policy training: ~1 min (5 epochs)
- Evaluation: ~2 min (50 episodes)
- **Total: ~8 minutes**

### Results
```
Strategy Comparison (50 episodes):
  Random:       a_ctrl* = 0.152 ± 0.031, MOCU = 0.046 ± 0.012
  Greedy MPNN:  a_ctrl* = 0.125 ± 0.023, MOCU = 0.032 ± 0.009 (↓18% / ↓30%)
  DAD:          a_ctrl* = 0.118 ± 0.020, MOCU = 0.028 ± 0.008 (↓22% / ↓39%)
```

## References

### Core Papers
- **Kuramoto Model**: Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence
- **MOCU Framework**: [Yoon et al. (2021) - Kuramoto OED Acceleration](https://github.com/bjyoontamu/Kuramoto-Model-OED-acceleration)
- **MPNN Surrogate**: [Sherry et al. (2023) - AccelerateOED](https://github.com/Levishery/AccelerateOED)
- **Deep Adaptive Design**: [Foster et al. (2021) - DAD](https://github.com/ae-foster/dad)

## License
MIT License - Educational and research purposes