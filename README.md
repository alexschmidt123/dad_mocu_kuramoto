# Deep Adaptive Design for Kuramoto Oscillator Networks

GPU-accelerated sequential experiment design for learning coupling structures in Kuramoto oscillator networks through active synchronization tests.

## Purpose & Innovation

### Problem Formulation
Given a network of coupled Kuramoto oscillators:

$$\dot{\theta}_i = \omega_i + \sum_{j} a_{ij}\sin(\theta_j - \theta_i) + a_{\text{ctrl}}\sin(\theta_c - \theta_i)$$

**Objective**: Minimize the pacemaker control parameter $a_{\text{ctrl}}^*$ needed to synchronize the network, under uncertainty about coupling strengths $a_{ij}$.

**Sequential Design**: Select the most informative pair-wise synchronization tests to efficiently reduce uncertainty about $\{a_{ij}\}$ and minimize $a_{\text{ctrl}}^*$.

### Mean Objective Cost of Uncertainty (MOCU)

We optimize experiment selection by minimizing expected regret under the current belief state:

$\text{MOCU}(h) = \mathbb{E}_{\theta|h}\left[C(a_{\text{IBR}}(h), \theta) - C(a^*(\theta), \theta)\right]$

where:
- $h$ represents current belief state (intervals for each $a_{ij}$)
- $a_{\text{IBR}}(h)$ is information-based robust control under belief $h$
- $a^*(\theta)$ is optimal control for true parameters $\theta = \{a_{ij}\}$
- $C(a, \theta)$ is control cost: $C(a, \theta) = a$ if system synchronizes, $\infty$ otherwise

**Expected Remaining MOCU (ERM)** for candidate experiment $\xi$:

$\text{ERM}(h, \xi) = \mathbb{E}_{y|\xi,h}\left[\text{MOCU}(h \oplus (\xi, y))\right]$

where $y \in \{\text{sync}, \text{not-sync}\}$ is the experiment outcome, and $h \oplus (\xi, y)$ is the updated belief.

### Key Contributions
1. **MPNN Surrogate**: Graph neural network approximates expensive MOCU/ERM computations via Monte Carlo estimation (1000× speedup)
2. **Deep Adaptive Design (DAD)**: Policy network trained via behavior cloning on greedy MPNN trajectories
3. **GPU Optimization**: Mixed precision training (AMP) + cuDNN optimization for RTX 4090 (~8 min full pipeline)

## Baseline Comparisons

| Strategy | Description | Reference |
|----------|-------------|-----------|
| **Random** | Uniformly sample candidate pairs at each step | Standard baseline |
| **Fixed Design** | Pre-determined non-adaptive experiment sequence (optimized offline) | [Foster et al. (2021)](https://github.com/ae-foster/dad) |
| **Greedy MPNN** | Minimize predicted ERM at each step using MPNN surrogate | [Sherry et al. (2023)](https://github.com/Levishery/AccelerateOED) |
| **DAD (Ours)** | Deep policy network trained via sPCE on adaptive trajectories | [Foster et al. (2021)](https://github.com/ae-foster/dad) |

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

## References

### Related Papers & Code

1. **MOCU Framework for Kuramoto Networks**  
   Yoon et al. (2021) - *Accelerating Optimal Experimental Design for Robust Synchronization of Uncertain Kuramoto Oscillator Model*  
   Repository: [github.com/bjyoontamu/Kuramoto-Model-OED-acceleration](https://github.com/bjyoontamu/Kuramoto-Model-OED-acceleration)

2. **MPNN Surrogate for OED Acceleration**  
   Sherry et al. (2023) - *Learning to Accelerate Optimal Experimental Design using Deep Learning*  
   Repository: [github.com/Levishery/AccelerateOED](https://github.com/Levishery/AccelerateOED)

3. **Deep Adaptive Design (DAD)**  
   Foster et al. (2021) - *Deep Adaptive Design: Amortizing Sequential Bayesian Experimental Design*  
   Repository: [github.com/ae-foster/dad](https://github.com/ae-foster/dad)

4. **Kuramoto Model**  
   Kuramoto, Y. (1984) - *Chemical Oscillations, Waves, and Turbulence*

## License
MIT License - Educational and research purposes