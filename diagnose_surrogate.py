#!/usr/bin/env python3
"""
Debug why surrogate training is producing constant predictions.
"""

import torch
import yaml
import numpy as np
import pickle

print("="*80)
print("SURROGATE TRAINING DIAGNOSTIC")
print("="*80)

# Load training data
cache_path = "dataset/train_N5_K4_n500_mc10_seed42.pkl"
print(f"\n1. Loading training data from {cache_path}")
with open(cache_path, 'rb') as f:
    train_data = pickle.load(f)

print(f"   ✓ Loaded {len(train_data)} samples")

# Extract MOCU labels
print("\n2. Analyzing MOCU labels in training data:")
all_mocu = []
for sample in train_data:
    # Intermediate MOCUs
    for step_data in sample['experiment_data']:
        all_mocu.append(step_data['mocu'])
    # Final MOCU
    all_mocu.append(sample['final_mocu'])

all_mocu = np.array(all_mocu)

print(f"   Total MOCU labels: {len(all_mocu)}")
print(f"   Mean: {all_mocu.mean():.4f}")
print(f"   Std:  {all_mocu.std():.4f}")
print(f"   Min:  {all_mocu.min():.4f}")
print(f"   Max:  {all_mocu.max():.4f}")
print(f"   Median: {np.median(all_mocu):.4f}")

# Check for issues
print("\n3. Checking for training data issues:")

if all_mocu.std() < 0.01:
    print("   ✗ PROBLEM: Labels have very low variance!")
    print("     → Model will learn to output constant (the mean)")
    print("     → Fix: Regenerate data with more diverse samples")
elif all_mocu.mean() > 5.0:
    print("   ✗ PROBLEM: Labels are extremely large!")
    print("     → Model may have numerical instability")
    print("     → Fix: Use label normalization or adjust mocu_scale")
elif all_mocu.min() < 0:
    print("   ✗ PROBLEM: Some labels are negative!")
    print("     → MOCU should always be non-negative")
    print("     → Fix: Debug MOCU computation")
else:
    print("   ✓ Label statistics look reasonable")

# Histogram
print("\n4. MOCU distribution:")
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 10.0]
hist, _ = np.histogram(all_mocu, bins=bins)
for i in range(len(bins)-1):
    bar = '█' * int(hist[i] / len(all_mocu) * 50)
    print(f"   [{bins[i]:.1f}, {bins[i+1]:.1f}): {bar} {hist[i]} ({hist[i]/len(all_mocu)*100:.1f}%)")

# Load trained model and check predictions
print("\n5. Loading trained surrogate model:")
from surrogate.mpnn_surrogate import MPNNSurrogate

cfg = yaml.safe_load(open('configs/config_fast.yaml'))

surrogate = MPNNSurrogate(
    mocu_scale=cfg["surrogate"].get("mocu_scale", 1.0),
    hidden=cfg["surrogate"]["hidden"],
    dropout=cfg["surrogate"]["dropout"]
)

try:
    surrogate.load_state_dict(torch.load('models/mpnn_surrogate.pth', 
                                         map_location='cpu', weights_only=True))
    surrogate.eval()
    print("   ✓ Model loaded")
except FileNotFoundError:
    print("   ✗ Model not found - run training first")
    exit(1)

print("\n6. Checking model predictions on training data:")
predictions = []

with torch.no_grad():
    for i in range(min(100, len(train_data))):
        sample = train_data[i]
        belief_graph = sample['final_belief_graph']
        pred = surrogate.forward_mocu(belief_graph).item()
        predictions.append(pred)

predictions = np.array(predictions)

print(f"   Mean prediction: {predictions.mean():.4f}")
print(f"   Std prediction:  {predictions.std():.4f}")
print(f"   Range: [{predictions.min():.4f}, {predictions.max():.4f}]")

if predictions.std() < 0.1:
    print("\n   ✗ CRITICAL PROBLEM: Model outputs nearly constant predictions!")
    print("     This means the model didn't learn anything meaningful.")
    print()
    print("   Possible causes:")
    print("     1. Training didn't converge (too few epochs)")
    print("     2. Learning rate too high (weights diverged)")
    print("     3. Learning rate too low (weights didn't update)")
    print("     4. Gradient vanishing (check activation functions)")
    print("     5. Model architecture too simple")
    print("     6. softplus + mocu_scale causing saturation")
else:
    print("   ✓ Model produces varied predictions")

# Check the specific issue: softplus saturation
print("\n7. Checking for softplus saturation:")
print("   In forward_mocu, output = softplus(head_output) * mocu_scale")
print(f"   If head outputs ~1.0, softplus(1.0) ≈ {torch.nn.functional.softplus(torch.tensor(1.0)).item():.4f}")
print(f"   With mocu_scale={cfg['surrogate'].get('mocu_scale', 1.0)}, final ≈ {torch.nn.functional.softplus(torch.tensor(1.0)).item() * cfg['surrogate'].get('mocu_scale', 1.0):.4f}")
print()
print("   This could explain constant ~0.88 output!")

print("\n8. DIAGNOSIS:")
print("   The model is outputting:")
print(f"     - Constant value near 0.88")
print(f"     - While labels are in range [{all_mocu.min():.2f}, {all_mocu.max():.2f}]")
print()
print("   ROOT CAUSE: Training loss minimized to predict the MEAN of labels")
print("   This happens when:")
print("     - Model can't learn the input→output mapping")
print("     - Training stopped too early")
print("     - Architecture is too simple for the task")

print("\n9. SOLUTION:")
print("   Option A (Quick fix): Normalize labels during training")
print("     - Divide all MOCU labels by their max (e.g., 2.0)")
print("     - Train model to predict normalized values")
print("     - Multiply predictions by max during inference")
print()
print("   Option B (Proper fix): Improve training")
print("     - Increase epochs from 50 → 200")
print("     - Monitor validation loss to ensure convergence")
print("     - Use learning rate scheduler")
print("     - Remove mocu_scale (just predict raw values)")
print()
print("   Option C (Nuclear option): Fix architecture")
print("     - Remove softplus (use raw output)")
print("     - Add batch normalization")
print("     - Increase model capacity")

print("\n" + "="*80)