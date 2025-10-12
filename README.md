
# DAD + MOCU Kuramoto (Fixed K) — Complete Demo

Paper-faithful demo for sequential experiment design on the **Kuramoto** model with the **single-scalar pacemaker control** (as in 2021/2023), using a lightweight **MPNN-like surrogate** and an optional **DAD** policy (encoder → aggregator → emitter).

- **Goal (fixed K):** run exactly `K` pair tests; update edge-interval beliefs; **after step K**, compute the **minimal pacemaker coupling** $a_\text{ctrl}^*$ (binary search) that synchronizes the **worst-case** matrix $A_{\min}$ built from post-test **lower bounds**.
- **Metrics:** **Primary** = terminal **MOCU**; **Co-primary** = final **$a_\text{ctrl}^*$**; Secondary = AUC of MOCU vs steps, decision latency.
- **Design choices:** Greedy ERM (2023-style surrogate baseline) or DAD (Deep Adaptive Design).

**References**  
• 2021 Kuramoto OED acceleration: https://github.com/bjyoontamu/Kuramoto-Model-OED-acceleration  
• 2023 MPNN OED acceleration: https://github.com/Levishery/AccelerateOED  
• Deep Adaptive Design (DAD): https://github.com/ae-foster/dad

## 🚀 Quick Start

### Installation

**Option 1: Using Conda (Recommended)**
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate dad_mocu_kuramoto
```

**Option 2: Using pip**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Demo

**Quick demo (single episodes):**
```bash
# Make sure to activate the environment first
conda activate dad_mocu_kuramoto  # or source venv/bin/activate for pip

python main_demo.py --mode quick
```

**Full evaluation (multiple episodes with statistics):**
```bash
conda activate dad_mocu_kuramoto
python main_demo.py --mode full --episodes 20 --verbose
```

**Force retrain surrogate model:**
```bash
conda activate dad_mocu_kuramoto
python main_demo.py --mode full --retrain
```

**Save results to file:**
```bash
conda activate dad_mocu_kuramoto
python main_demo.py --mode full --episodes 10 --save-results results.json
```

### Command Line Options

- `--config`: Path to configuration file (default: `configs/exp_fixedK.yaml`)
- `--mode`: Demo mode - `quick` for single episodes, `full` for comprehensive evaluation
- `--episodes`: Number of episodes for full evaluation (default: 10)
- `--retrain`: Force retrain the surrogate model
- `--save-results`: Path to save results JSON file
- `--verbose`: Enable verbose output

## 📊 What This Demo Does

This demo implements a complete pipeline for sequential experiment design on the Kuramoto oscillator model:

1. **Data Generation**: Creates synthetic training data for the surrogate model
2. **Surrogate Training**: Trains an MPNN to predict MOCU, ERM, and synchronization
3. **Strategy Comparison**: Compares different experiment selection strategies:
   - **Random**: Randomly selects pairs to test
   - **Greedy ERM**: Uses the surrogate to greedily minimize expected remaining MOCU
   - **DAD**: Deep Adaptive Design policy trained via behavior cloning
4. **Evaluation**: Runs multiple episodes and generates comprehensive statistics and plots

## 🏗️ Project Structure

```
dad_mocu_kuramoto/
├── configs/
│   └── exp_fixedK.yaml          # Configuration file
├── core/                        # Core simulation components
│   ├── belief.py               # Belief state management
│   ├── kuramoto_env.py         # Environment for pair testing
│   ├── pacemaker_control.py    # Kuramoto + pacemaker simulation
│   └── bisection.py            # Binary search for optimal control
├── data_generation/            # Synthetic data generation
│   └── synthetic_data.py       # Training data generator
├── surrogate/                  # Neural network surrogate models
│   ├── mpnn_surrogate.py       # MPNN architecture
│   └── train_surrogate.py      # Training pipeline
├── design/                     # Experiment design strategies
│   ├── greedy_erm.py           # Greedy ERM baseline
│   ├── dad_policy.py           # DAD policy architecture
│   └── train_bc.py             # Behavior cloning training
├── eval/                       # Evaluation and metrics
│   ├── run_eval.py             # Episode execution
│   └── metrics.py              # Statistics and plotting
├── utils/                      # Utility functions
│   └── graph_ops.py            # Graph operations
├── main_demo.py                # Main demo script
└── requirements.txt            # Dependencies
```

## 🔧 Configuration

The demo is configured via `configs/exp_fixedK.yaml`:

- `N`: Number of oscillators (default: 5)
- `K`: Number of experiments to run (default: 4)
- `surrogate`: Training parameters for the neural network
- `dad_bc`: Behavior cloning parameters for DAD policy
- `sim`: Simulation parameters for Kuramoto model

## 📈 Output

The demo generates:

1. **Console output**: Real-time progress and final statistics
2. **Plots**: 
   - MOCU evolution curves (`mocu_curves.png`)
   - Control parameter distributions (`a_ctrl_distribution.png`)
3. **Results file**: JSON with detailed results (if `--save-results` specified)

## 🧪 Example Output

```
================================================================================
COMPREHENSIVE EVALUATION
================================================================================
Loading pre-trained surrogate from trained_surrogate.pth
Training DAD policy...
[BC] epoch 1: avg loss = 0.1234
[BC] epoch 2: avg loss = 0.0987
[BC] epoch 3: avg loss = 0.0876

Running evaluation with 10 episodes per strategy...

================================================================================
STRATEGY COMPARISON RESULTS
================================================================================

Random:
--------
  a_ctrl_star:
    Mean: 0.1234 ± 0.0234
    Range: [0.0987, 0.1567]
    Median: 0.1201
  terminal_mocu:
    Mean: 0.0456 ± 0.0123
    Range: [0.0234, 0.0678]
    Median: 0.0432
  Episodes: 10
  Avg Time: 0.1234s

GreedyERM:
----------
  a_ctrl_star:
    Mean: 0.0987 ± 0.0156
    Range: [0.0765, 0.1234]
    Median: 0.0954
  terminal_mocu:
    Mean: 0.0321 ± 0.0089
    Range: [0.0156, 0.0456]
    Median: 0.0298
  Episodes: 10
  Avg Time: 0.1567s
```

---

## Method (slide-faithful, concise)

### (1) Problem & control (paper-accurate, fixed K)

**Kuramoto with pacemaker**
$$
\dot{\theta}_i(t)
=\omega_i+\sum_{j\neq i} a_{ij}\,\sin\!\big(\theta_j(t)-\theta_i(t)\big)
\;+\;a_\text{ctrl}\,\sin\!\big(\theta_c(t)-\theta_i(t)\big),
\quad i=1,\dots,N.
$$

- **Unknowns:** symmetric couplings $a_{ij}\ge 0$. **Known:** $\{\omega_i\}_{i=1}^N$.  
- **Belief:** for each edge $(i,j)$, an interval $[a^\ell_{ij},\,a^u_{ij}]$ (uniform working prior).  
- **Design primitive (pair test):** choose $\xi=(i,j)$, run a short window, observe $y\in\{\text{sync},\text{not}\}$.  
- **Design goal (fixed K):** after K tests, choose **one** scalar $a_\text{ctrl}^*$ (pacemaker-to-all) that synchronizes the **worst-case** $A_{\min}=(a^\ell_{ij})$.

### (2) Pair-test model & belief update

**Two-oscillator lock threshold**
$$
\lambda_{ij}=\tfrac{1}{2}\,|\omega_i-\omega_j|.
$$

**Predictive (uniform on interval)**
$$
p\!\left(\text{sync}\mid h,\xi=(i,j)\right)
=\frac{\max\!\left(0,\;a^u_{ij}-\tilde a_{ij}\right)}{a^u_{ij}-a^\ell_{ij}},
\qquad
\tilde a_{ij}=\min\!\left\{\max\!\left(\lambda_{ij},a^\ell_{ij}\right),\,a^u_{ij}\right\}.
$$

**Interval update (noise-free rule-of-thumb)**
$$
y=\text{sync}\;\Rightarrow\; a^\ell_{ij}\leftarrow \max\!\big(a^\ell_{ij},\lambda_{ij}\big),
\qquad
y=\text{not}\;\Rightarrow\; a^u_{ij}\leftarrow \min\!\big(a^u_{ij},\lambda_{ij}\big).
$$

### (3) MOCU, IBR control, ERM

Let the control cost be $C(a_\text{ctrl},\theta)$ (monotone in $a_\text{ctrl}$), with $\theta=\{a_{ij}\}$.

- **Clairvoyant control:** $a^{*}(\theta)=\arg\min_{a\ge 0} C(a,\theta)$.  
- **IBR (robust) control:** $a_\text{IBR}(h)=\arg\min_{a\ge 0}\mathbb{E}_{\theta\mid h} C(a,\theta)$.  
- **MOCU:**
$$
\text{MOCU}(h)=\mathbb{E}_{\theta\mid h}\!\Big[C\big(a_\text{IBR}(h),\theta\big)-C\big(a^{*}(\theta),\theta\big)\Big].
$$
- **ERM (Expected Remaining MOCU) for candidate $\xi$:**
$$
\text{ERM}(h,\xi)=\mathbb{E}_{y\mid h,\xi}\big[\text{MOCU}(h\oplus(\xi,y))\big].
$$

**Design target:** choose $\xi_{1:K}$ to minimize terminal $\text{MOCU}(h_K)$ (typically also reducing final $a_\text{ctrl}^*(h_K)$).

### (4) 2023-style MPNN surrogate (fast utilities & sync check)

- **Inputs:** belief-graph features (node $\omega_i$; edge $[a^\ell,a^u]$, width, $\lambda_{ij}$, tested flag), candidate $\xi$.  
- **Outputs:** $\widehat{\text{MOCU}}(h)$, $\widehat{\text{ERM}}(h,\xi)$, and $\widehat{\text{Sync}}(A,u)$ for fast bisection.  
- **Training (offline):** regress to slow labels (MC over $\theta$ + ODE + binary search in $a_\text{ctrl}$); then **freeze**.

### (5) DAD policy (pair selection)

Policy $\pi_\phi(\xi\mid h)$: **MPNN encoder** → **history aggregator** (LSTM or SUM over $(\xi,y)$) → **emitter** scoring pairs.  
**Training (offline):** **imitation** of greedy $\arg\min_\xi \widehat{\text{ERM}}(h,\xi)$ (warm start), optional **RL** at fixed horizon $K$ with reward $-\widehat{\text{MOCU}}(h_K)$ (and/or step $\Delta \widehat{\text{MOCU}}$).

### (6) Final control $a_\text{ctrl}^*$ (paper-faithful)

- Build worst-case matrix $A_{\min}=(a^\ell_{ij})$ after K tests.  
- Binary search for $a_\text{ctrl}^*$ using a sync check (Kuramoto + pacemaker ODE; criterion: order parameter $R(t)$ above threshold after burn-in) or the surrogate’s $\widehat{\text{Sync}}$ head.

---

## Repository layout (tree + roles)

```text
dad_mocu_kuramoto_demo/
├─ configs/
│  └─ exp_fixedK.yaml
├─ core/
│  ├─ belief.py               # History h_k; [a^ℓ,a^u] per edge; λ_ij; belief-graph features
│  ├─ kuramoto_env.py         # Pair-test env: run test on (i,j), update interval, expose candidates
│  ├─ pacemaker_control.py    # Kuramoto + pacemaker ODE; order parameter R(t); sync checker
│  └─ bisection.py            # Binary search for minimal a_ctrl* that passes sync on A_min
├─ surrogate/
│  └─ mpnn_surrogate.py       # Demo MPNN-like surrogate: MOCU_hat, ERM_hat, Sync_hat
├─ design/
│  ├─ greedy_erm.py           # Paper baseline: argmin_xi ERM_hat(h, xi)
│  ├─ dad_policy.py           # DAD policy: encoder + (LSTM/SUM) aggregator + emitter
│  └─ train_bc.py             # Behavior cloning of greedy-ERM teacher (fixed-K episodes)
├─ eval/
│  ├─ run_eval.py             # Run one episode; report a_ctrl* and terminal MOCU_hat
│  └─ metrics.py              # Print/format helpers
├─ utils/
│  └─ graph_ops.py            # Small graph utilities (e.g., enumerate pairs)
├─ main_demo.py               # Quickstart: Greedy ERM vs Random
└─ requirements.txt
