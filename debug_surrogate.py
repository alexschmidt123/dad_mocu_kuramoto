#!/usr/bin/env python3
"""
Debug script to investigate surrogate model learning issues.
"""

import yaml
import numpy as np
import torch
import pickle
from surrogate.mpnn_surrogate import MPNNSurrogate
from core.kuramoto_env import PairTestEnv

def load_config(path: str):
    """Load configuration."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def debug_surrogate_predictions(surrogate, belief_graphs, label="Debug"):
    """Debug surrogate model predictions on multiple belief graphs."""
    print(f"\n{label} - Surrogate Model Predictions:")
    print("="*50)
    
    predictions = []
    for i, belief_graph in enumerate(belief_graphs[:5]):  # Check first 5
        try:
            mocu_pred = surrogate.forward_mocu(belief_graph)
            pred_value = mocu_pred.item()
            predictions.append(pred_value)
            print(f"  Belief {i+1}: MOCU = {pred_value:.6f}")
        except Exception as e:
            print(f"  Belief {i+1}: ERROR - {e}")
    
    if predictions:
        print(f"\nPrediction Statistics:")
        print(f"  Mean: {np.mean(predictions):.6f}")
        print(f"  Std:  {np.std(predictions):.6f}")
        print(f"  Min:  {np.min(predictions):.6f}")
        print(f"  Max:  {np.max(predictions):.6f}")
        print(f"  Unique values: {len(set(predictions))}")
    
    return predictions

def debug_training_data():
    """Debug training data diversity."""
    print("\n" + "="*80)
    print("TRAINING DATA ANALYSIS")
    print("="*80)
    
    # Load training data
    try:
        with open('dataset/train_N5_K4_n500_mc10_seed42.pkl', 'rb') as f:
            train_data = pickle.load(f)
        print(f"✓ Loaded training data: {len(train_data)} samples")
    except FileNotFoundError:
        print("❌ Training data not found!")
        return None, None
    
    # Analyze MOCU values
    mocu_values = []
    belief_graphs = []
    
    for sample in train_data:
        # Final MOCU
        mocu_values.append(sample['final_mocu'])
        belief_graphs.append(sample['final_belief_graph'])
        
        # Intermediate MOCU values
        for step in sample['experiment_data']:
            mocu_values.append(step['mocu'])
            belief_graphs.append(step['belief_graph'])
    
    print(f"\nMOCU Value Analysis:")
    print(f"  Total MOCU values: {len(mocu_values)}")
    print(f"  Mean: {np.mean(mocu_values):.6f}")
    print(f"  Std:  {np.std(mocu_values):.6f}")
    print(f"  Min:  {np.min(mocu_values):.6f}")
    print(f"  Max:  {np.max(mocu_values):.6f}")
    print(f"  Unique values: {len(set(mocu_values))}")
    
    # Check if MOCU values are too similar
    if np.std(mocu_values) < 1e-6:
        print("⚠️  WARNING: MOCU values have very low variance!")
        print("   This could explain why the model always predicts the same value.")
    
    return mocu_values, belief_graphs

def debug_model_weights(surrogate):
    """Debug model weights to check if they're learning."""
    print("\n" + "="*80)
    print("MODEL WEIGHTS ANALYSIS")
    print("="*80)
    
    total_params = 0
    zero_params = 0
    identical_layers = 0
    
    for name, param in surrogate.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
            
            # Check if all weights in a layer are identical
            if param.numel() > 1:
                unique_vals = torch.unique(param).numel()
                if unique_vals == 1:
                    identical_layers += 1
                    print(f"⚠️  {name}: All weights identical ({param[0].item():.6f})")
    
    print(f"\nWeight Statistics:")
    print(f"  Total parameters: {total_params}")
    print(f"  Zero parameters: {zero_params} ({100*zero_params/total_params:.1f}%)")
    print(f"  Identical layers: {identical_layers}")
    
    if zero_params / total_params > 0.5:
        print("⚠️  WARNING: Too many zero parameters - model might not be learning!")
    
    if identical_layers > 0:
        print("⚠️  WARNING: Some layers have identical weights - model might be stuck!")

def debug_erm_predictions(surrogate, belief_graphs):
    """Debug ERM predictions for different candidate pairs."""
    print("\n" + "="*80)
    print("ERM PREDICTIONS ANALYSIS")
    print("="*80)
    
    if not belief_graphs:
        print("No belief graphs available for ERM testing")
        return
    
    belief_graph = belief_graphs[0]  # Use first belief graph
    candidates = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]  # Sample pairs
    
    print(f"Testing ERM predictions for {len(candidates)} candidate pairs:")
    erm_predictions = []
    
    for i, (cand_i, cand_j) in enumerate(candidates):
        try:
            erm_pred = surrogate.forward_erm(belief_graph, (cand_i, cand_j))
            pred_value = erm_pred.item()
            erm_predictions.append(pred_value)
            print(f"  Pair ({cand_i}, {cand_j}): ERM = {pred_value:.6f}")
        except Exception as e:
            print(f"  Pair ({cand_i}, {cand_j}): ERROR - {e}")
    
    if erm_predictions:
        print(f"\nERM Prediction Statistics:")
        print(f"  Mean: {np.mean(erm_predictions):.6f}")
        print(f"  Std:  {np.std(erm_predictions):.6f}")
        print(f"  Min:  {np.min(erm_predictions):.6f}")
        print(f"  Max:  {np.max(erm_predictions):.6f}")
        print(f"  Unique values: {len(set(erm_predictions))}")

def main():
    print("="*80)
    print("SURROGATE MODEL DEBUG ANALYSIS")
    print("="*80)
    
    # Load config
    cfg = load_config('configs/config_fast.yaml')
    
    # Debug training data
    mocu_values, belief_graphs = debug_training_data()
    
    if belief_graphs is None:
        print("Cannot proceed without training data")
        return
    
    # Load surrogate model
    try:
        print("\n" + "="*80)
        print("LOADING SURROGATE MODEL")
        print("="*80)
        
        surrogate = MPNNSurrogate(
            mocu_scale=cfg["surrogate"].get("mocu_scale", 1.0),
            hidden=cfg["surrogate"]["hidden"],
            dropout=cfg["surrogate"]["dropout"]
        )
        
        state_dict = torch.load('models/mpnn_surrogate.pth', map_location='cpu', weights_only=True)
        surrogate.load_state_dict(state_dict)
        surrogate.eval()
        
        print("✓ Surrogate model loaded successfully")
        
    except Exception as e:
        print(f"❌ Failed to load surrogate model: {e}")
        return
    
    # Debug model predictions
    debug_surrogate_predictions(surrogate, belief_graphs, "MOCU Predictions")
    
    # Debug model weights
    debug_model_weights(surrogate)
    
    # Debug ERM predictions
    debug_erm_predictions(surrogate, belief_graphs)
    
    print("\n" + "="*80)
    print("DEBUG ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
