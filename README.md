# Objective-Based Deep Adaptive Design for Kuramoto Oscillator Networks

GPU-accelerated sequential experiment design for learning coupling structures in Kuramoto oscillator networks through active synchronization tests.

## Purpose & Innovation

### Problem Formulation
Given a network of coupled Kuramoto oscillators:

$$\dot{\theta}_i = \omega_i + \sum_{j} a_{ij}\sin(\theta_j - \theta_i) + a_{\text{ctrl}}\sin(\theta_c - \theta_i)$$

**Objective**: Minimize the pacemaker control parameter $a_{\text{ctrl}}^*$ needed to synchronize the network, under uncertainty about coupling strengths $a_{ij}$.

**Sequential Design**: Select the most informative pair-wise synchronization tests to efficiently reduce uncertainty about $\{a_{ij}\}$ and minimize $a_{\text{ctrl}}^*$.

### Mean Objective Cost of Uncertainty (MOCU)

We optimize experiment selection by minimizing expected regret under the current belief state:

$$\text{MOCU}(h) = \mathbb{E}_{\theta|h}\left[C(a_{\text{IBR}}(h), \theta) - C(a^*(\theta), \theta)\right]$$

**Expected Remaining MOCU (ERM)** for candidate experiment $\xi$:

$$\text{ERM}(h, \xi) = \mathbb{E}_{y|\xi,h}\left[\text{MOCU}(h \oplus (\xi, y))\right]$$

### Key Contributions
1. **MPNN Surrogate**: Graph neural network approximates expensive MOCU/ERM computations (1000× speedup)
2. **Fixed Design**: Pre-optimized static sequence using MPNN greedy selection
3. **DAD Policy**: Deep policy network trained via behavior cloning on MPNN trajectories
4. **GPU Optimization**: Mixed precision training + adaptive ODE integration (RK45)

## Baseline Comparisons

| Strategy | Description | Components Used |
|----------|-------------|-----------------|
| **Random** | Uniformly sample candidate pairs at each step | None |
| **Fixed Design** | Static sequence optimized offline using MPNN greedy | MPNN Surrogate |
| **Greedy MPNN** | Minimize predicted ERM at each step | MPNN Surrogate |
| **DAD with MOCU** | Adaptive policy trained on MPNN trajectories | MPNN Surrogate + Policy Network |

**Note**: 
- Fixed Design is derived from DAD framework using greedy MPNN selection
- DAD contains both MPNN surrogate and learned policy components
- All adaptive methods (Fixed, Greedy, DAD) depend on the MPNN surrogate

## Repository Structure

```
dad_mocu_kuramoto/
├── generate_data.py         # Step 1: Generate training data
├── train.py                 # Step 2: Train models
├── test.py                  # Step 3: Evaluate models
├── configs/
│   ├── config.yaml         # Full configuration
│   └── config_fast.yaml    # Fast configuration (for testing)
├── dataset/                # Generated training data (cached)
├── models/                 # Trained models
│   ├── mpnn_surrogate.pth
│   ├── fixed_design.pkl
│   └── dad_policy.pth
├── core/                   # Simulation engine
│   ├── belief.py          # Bayesian belief updates
│   ├── kuramoto_env.py    # Experiment environment
│   ├── pacemaker_control.py   # ODE integration (RK45)
│   └── bisection.py       # Binary search for optimal control
├── surrogate/             # MPNN surrogate model
│   ├── mpnn_surrogate.py
│   └── train_surrogate.py
├── design/                # Experiment selection strategies
│   ├── greedy_erm.py     # Greedy baseline
│   ├── dad_policy.py     # Deep adaptive design
│   └── train_rl.py       # rl training
└── eval/                  # Evaluation utilities
    ├── run_eval.py
    └── metrics.py
```

## Installation & Reproduction

### Prerequisites
- Python 3.9+
- CUDA 12.1+ (for GPU training)
- 24-core CPU recommended (for parallel data generation)

### Setup
```bash
# Clone repository
git clone [<repo-url>](https://github.com/alexschmidt123/dad_mocu_kuramoto)
cd dad_mocu_kuramoto

# Install PyTorch with CUDA support (optimized for RTX 4090)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install numpy scipy pyyaml matplotlib tqdm
```

### Quick Start (Fast Config, ~2 hours total)

Optimized for 24-core CPU + RTX 4090 GPU:

```bash
# Step 1: Generate training data (~1.5 hours with 20 workers)
python generate_data.py --config configs/config_fast.yaml --split both --parallel --workers 20

# Step 2: Train all models (~10 min on GPU)
python train.py --config configs/config_fast.yaml --methods all

# Step 3: Evaluate (~2 min)
python test.py --config configs/config_fast.yaml --episodes 50
```

**Fast Config Parameters:**
- Training samples: 500
- Validation samples: 100
- MC samples: 10
- Simulation time: 3.0s
- Epochs: 50

### Production Run (Full Config, ~15 hours total)

For best quality results:

```bash
# Step 1: Generate training data (~12-15 hours with 20 workers)
python generate_data.py --config configs/config.yaml --split both --parallel --workers 20

# Step 2: Train all models (~20 min on GPU)
python train.py --config configs/config.yaml --methods all

# Step 3: Evaluate (~10 min)
python test.py --config configs/config.yaml --episodes 100 --save-results results.json
```

**Full Config Parameters:**
- Training samples: 2000
- Validation samples: 400
- MC samples: 30
- Simulation time: 5.0s
- Epochs: 100


## References

- [Kuramoto Model OED Acceleration](https://github.com/bjyoontamu/Kuramoto-Model-OED-acceleration) - Original implementation by Yoon et al.
- [Deep Adaptive Design](https://github.com/ae-foster/dad) - DAD framework by Foster et al.
- [Accelerate OED](https://github.com/Levishery/AccelerateOED) - MPNN acceleration by Chen et al.

