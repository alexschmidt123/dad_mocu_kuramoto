# Objective-Based Deep Adaptive Design for Kuramoto Oscillator Networks

GPU-accelerated sequential experiment design for learning coupling structures in Kuramoto oscillator networks through active synchronization tests.

## Problem Formulation

Given a network of coupled Kuramoto oscillators:

$$\dot{\theta}_i = \omega_i + \sum_{j} a_{ij}\sin(\theta_j - \theta_i) + a_{\text{ctrl}}\sin(\theta_c - \theta_i)$$

**Objective**: Minimize the pacemaker control parameter $a_{\text{ctrl}}^*$ needed to synchronize the network, under uncertainty about coupling strengths $a_{ij}$.

**Sequential Design**: Run exactly K pair-wise synchronization tests, update belief intervals based on sync/no-sync outcomes, and select each next pair by minimizing the predicted Expected Remaining MOCU (Mean Objective Cost of Uncertainty).

### Mean Objective Cost of Uncertainty (MOCU)

MOCU quantifies the expected regret of acting under uncertainty:

$$\text{MOCU}(h) = \mathbb{E}_{\theta|h}\left[C(a_{\text{IBR}}(h), \theta) - C(a^*(\theta), \theta)\right]$$

where:
- $a_{\text{IBR}}(h)$ is the worst-case control needed given current belief $h$
- $a^*(\theta)$ is the optimal control if true couplings $\theta$ were known

**Expected Remaining MOCU (ERM)** for candidate experiment $\xi$:

$$\text{ERM}(h, \xi) = \mathbb{E}_{y|\xi,h}\left[\text{MOCU}(h \oplus (\xi, y))\right]$$

## Key Contributions

1. **MPNN Surrogate (2023)**: Message-passing neural network approximates expensive MOCU/ERM computations via Monte Carlo sampling (~1000× speedup over exact computation)

2. **Fixed Design**: Pre-optimized static K-step sequence using offline greedy MPNN-based selection

3. **DAD Policy**: Deep adaptive policy trained via reinforcement learning to minimize final MOCU by sequentially selecting optimal pair tests

4. **GPU Acceleration**: PyCUDA-based batch ODE integration for parallel Kuramoto simulation (10-50× speedup for data generation)

## Experiment Selection Strategies

| Strategy | Description | Selection Method |
|----------|-------------|------------------|
| **Random** | Uniformly sample untested pairs | Random sampling |
| **Fixed Design** | Static K-step sequence optimized offline | Greedy MPNN (offline) |
| **Greedy MPNN** | Myopic minimization at each step | $\arg\min_\xi \text{ERM}_{\text{MPNN}}(h, \xi)$ |
| **DAD Policy** | Adaptive policy network | $\pi_\theta(h, \text{candidates})$ trained via RL |

**Key differences**:
- **Fixed Design**: Computed once offline, same sequence for all problem instances
- **Greedy MPNN**: Adapts to observations but makes myopic (one-step-ahead) decisions
- **DAD Policy**: Learns to anticipate future steps, trained to minimize terminal MOCU

**Dependencies**:
- All adaptive methods require the MPNN surrogate for fast MOCU/ERM prediction
- DAD Policy additionally uses a learned policy network trained on MPNN-guided trajectories

## Repository Structure

```
dad_mocu_kuramoto/
├── generate_data_gpu.py     # GPU-accelerated data generation
├── train.py                 # Train all models
├── test.py                  # Evaluate trained models
├── configs/
│   ├── config.yaml         # Full configuration
│   └── config_fast.yaml    # Fast configuration (testing)
├── dataset/                # Generated training data (cached)
├── models/                 # Trained models
│   ├── mpnn_surrogate.pth
│   ├── fixed_design.pkl
│   └── dad_policy.pth
├── core/                   # Simulation engine
│   ├── belief.py          # Bayesian belief updates
│   ├── kuramoto_env.py    # Experiment environment
│   ├── pacemaker_control.py   # GPU/CPU ODE integration
│   └── bisection.py       # Binary search for optimal control
├── surrogate/             # MPNN surrogate model
│   ├── mpnn_surrogate.py
│   └── train_surrogate.py
├── design/                # Experiment selection strategies
│   ├── greedy_erm.py     # Greedy MPNN baseline
│   ├── dad_policy.py     # DAD policy network
│   └── train_rl.py       # RL training
└── eval/                  # Evaluation utilities
    ├── run_eval.py
    └── metrics.py
```

## Installation & Reproduction

```bash
# Clone repository
git clone https://github.com/alexschmidt123/dad_mocu_kuramoto
cd dad_mocu_kuramoto

# Create conda environment
conda create -n dad_mocu_kuramoto python=3.9
conda activate dad_mocu_kuramoto

# Install CUDA toolkit and PyCUDA via conda
conda install -c nvidia cuda-toolkit
pip install pycuda

# Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies (includes PyCUDA)
pip install -r requirements.txt
```

### Quick Start (Fast Config, ~30 minutes)

```bash
# Step 1: Generate training data (~15 min)
python generate_data.py --config configs/config_fast.yaml --split both

# Step 2: Train all models (~10 min)
python train.py --config configs/config_fast.yaml --methods all

# Step 3: Evaluate (~5 min)
python test.py --config configs/config_fast.yaml --episodes 50
```

### Full Configuration (~2-3 hours)

```bash
# Step 1: Generate training data (~1.5 hours)
python generate_data.py --config configs/config.yaml --split both

# Step 2: Train all models (~20 min)
python train.py --config configs/config.yaml --methods all

# Step 3: Evaluate (~10 min)
python test.py --config configs/config.yaml --episodes 100 --save-results results.json
```


## References

- [Kuramoto Model OED Acceleration](https://github.com/bjyoontamu/Kuramoto-Model-OED-acceleration) - Original implementation by Yoon et al.
- [Deep Adaptive Design](https://github.com/ae-foster/dad) - DAD framework by Foster et al.
- [Accelerate OED](https://github.com/Levishery/AccelerateOED) - MPNN acceleration by Chen et al.
