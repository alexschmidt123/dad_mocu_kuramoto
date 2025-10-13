"""
Fixed training pipeline for the MPNN surrogate model.
Properly handles graph data and trains all prediction heads.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from typing import List, Dict, Tuple
import os
from surrogate.mpnn_surrogate import MPNNSurrogate
from data_generation.synthetic_data import SyntheticDataGenerator


class SurrogateTrainer:
    """Trainer for the MPNN surrogate model with proper batching."""
    
    def __init__(self, model: MPNNSurrogate, device: str = None):
        self.model = model.to(device)
        self.device = device
        
    def prepare_mocu_data(self, dataset: List[Dict]) -> List[Tuple]:
        """Prepare MOCU training data (belief_graph, label)."""
        data = []
        for exp_data in dataset:
            # Use both intermediate and final states
            for step_data in exp_data['experiment_data']:
                data.append((step_data['belief_graph'], step_data['mocu']))
            # Add final state
            data.append((exp_data['final_belief_graph'], exp_data['final_mocu']))
        return data
    
    def prepare_erm_data(self, dataset: List[Dict]) -> List[Tuple]:
        """Prepare ERM training data (belief_graph, candidate_pair, label)."""
        data = []
        for exp_data in dataset:
            for step_data in exp_data['experiment_data']:
                belief_graph = step_data['belief_graph']
                for xi, erm_score in step_data['erm_scores'].items():
                    data.append((belief_graph, xi, erm_score))
        return data
    
    def prepare_sync_data(self, dataset: List[Dict]) -> List[Tuple]:
        """Prepare Sync training data (belief_graph, A_min, a_ctrl, label)."""
        data = []
        for exp_data in dataset:
            belief_graph = exp_data['final_belief_graph']
            # Extract A_min from belief graph edges (lower bounds)
            # We'll compute it from the experiment data
            omega = exp_data['omega']
            
            for a_ctrl, sync_label in exp_data['sync_scores'].items():
                # We need to pass A_min, but we'll encode it in the model differently
                # For now, pass omega and let the model learn from belief_graph
                data.append((belief_graph, omega, a_ctrl, sync_label))
        return data
    
    def train_mocu(self, train_data: List[Dict], val_data: List[Dict] = None,
                   epochs: int = 50, lr: float = 1e-3, batch_size: int = 32):
        """Train MOCU prediction head."""
        print("Training MOCU prediction head...")
        
        train_samples = self.prepare_mocu_data(train_data)
        val_samples = self.prepare_mocu_data(val_data) if val_data else None
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            
            # Shuffle training data
            indices = np.random.permutation(len(train_samples))
            
            for batch_start in range(0, len(train_samples), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_samples = [train_samples[i] for i in batch_indices]
                
                optimizer.zero_grad()
                
                # Process each sample in batch (graphs can't be easily batched)
                predictions = []
                labels = []
                
                for belief_graph, mocu_label in batch_samples:
                    pred = self.model.forward_mocu(belief_graph)
                    predictions.append(pred)
                    labels.append(mocu_label)
                
                predictions = torch.cat(predictions)
                labels = torch.tensor(labels, dtype=torch.float32, device=self.device)
                
                loss = criterion(predictions.squeeze(), labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = epoch_loss / n_batches
            train_losses.append(avg_train_loss)
            
            # Validation
            if val_samples:
                self.model.eval()
                val_loss = 0.0
                n_val_batches = 0
                
                with torch.no_grad():
                    for batch_start in range(0, len(val_samples), batch_size):
                        batch_samples = val_samples[batch_start:batch_start + batch_size]
                        
                        predictions = []
                        labels = []
                        
                        for belief_graph, mocu_label in batch_samples:
                            pred = self.model.forward_mocu(belief_graph)
                            predictions.append(pred)
                            labels.append(mocu_label)
                        
                        predictions = torch.cat(predictions)
                        labels = torch.tensor(labels, dtype=torch.float32, device=self.device)
                        
                        loss = criterion(predictions.squeeze(), labels)
                        val_loss += loss.item()
                        n_val_batches += 1
                
                avg_val_loss = val_loss / n_val_batches
                val_losses.append(avg_val_loss)
            
            if (epoch + 1) % 10 == 0:
                val_str = f", Val Loss = {val_losses[-1]:.4f}" if val_losses else ""
                print(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.4f}{val_str}")
        
        return {'train_losses': train_losses, 'val_losses': val_losses}
    
    def train_erm(self, train_data: List[Dict], val_data: List[Dict] = None,
                  epochs: int = 50, lr: float = 1e-3, batch_size: int = 32):
        """Train ERM prediction head."""
        print("Training ERM prediction head...")
        
        train_samples = self.prepare_erm_data(train_data)
        val_samples = self.prepare_erm_data(val_data) if val_data else None
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            
            indices = np.random.permutation(len(train_samples))
            
            for batch_start in range(0, len(train_samples), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_samples = [train_samples[i] for i in batch_indices]
                
                optimizer.zero_grad()
                
                predictions = []
                labels = []
                
                for belief_graph, xi, erm_label in batch_samples:
                    pred = self.model.forward_erm(belief_graph, xi)
                    predictions.append(pred)
                    labels.append(erm_label)
                
                predictions = torch.cat(predictions)
                labels = torch.tensor(labels, dtype=torch.float32, device=self.device)
                
                loss = criterion(predictions.squeeze(), labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = epoch_loss / n_batches
            train_losses.append(avg_train_loss)
            
            # Validation
            if val_samples:
                self.model.eval()
                val_loss = 0.0
                n_val_batches = 0
                
                with torch.no_grad():
                    for batch_start in range(0, len(val_samples), batch_size):
                        batch_samples = val_samples[batch_start:batch_start + batch_size]
                        
                        predictions = []
                        labels = []
                        
                        for belief_graph, xi, erm_label in batch_samples:
                            pred = self.model.forward_erm(belief_graph, xi)
                            predictions.append(pred)
                            labels.append(erm_label)
                        
                        predictions = torch.cat(predictions)
                        labels = torch.tensor(labels, dtype=torch.float32, device=self.device)
                        
                        loss = criterion(predictions.squeeze(), labels)
                        val_loss += loss.item()
                        n_val_batches += 1
                
                avg_val_loss = val_loss / n_val_batches
                val_losses.append(avg_val_loss)
            
            if (epoch + 1) % 10 == 0:
                val_str = f", Val Loss = {val_losses[-1]:.4f}" if val_losses else ""
                print(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.4f}{val_str}")
        
        return {'train_losses': train_losses, 'val_losses': val_losses}
    
    def train_sync(self, train_data: List[Dict], val_data: List[Dict] = None,
                   epochs: int = 50, lr: float = 1e-3, batch_size: int = 32):
        """Train Sync prediction head."""
        print("Training Sync prediction head...")
        
        train_samples = self.prepare_sync_data(train_data)
        val_samples = self.prepare_sync_data(val_data) if val_data else None
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            
            indices = np.random.permutation(len(train_samples))
            
            for batch_start in range(0, len(train_samples), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_samples = [train_samples[i] for i in batch_indices]
                
                optimizer.zero_grad()
                
                predictions = []
                labels = []
                
                for belief_graph, omega, a_ctrl, sync_label in batch_samples:
                    # Create dummy A_min (will be ignored in current implementation)
                    A_min = np.zeros((len(omega), len(omega)))
                    pred = self.model.forward_sync(A_min, a_ctrl, belief_graph)
                    predictions.append(pred)
                    labels.append(sync_label)
                
                predictions = torch.cat(predictions)
                labels = torch.tensor(labels, dtype=torch.float32, device=self.device)
                
                loss = criterion(predictions.squeeze(), labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = epoch_loss / n_batches
            train_losses.append(avg_train_loss)
            
            # Validation
            if val_samples:
                self.model.eval()
                val_loss = 0.0
                n_val_batches = 0
                
                with torch.no_grad():
                    for batch_start in range(0, len(val_samples), batch_size):
                        batch_samples = val_samples[batch_start:batch_start + batch_size]
                        
                        predictions = []
                        labels = []
                        
                        for belief_graph, omega, a_ctrl, sync_label in batch_samples:
                            A_min = np.zeros((len(omega), len(omega)))
                            pred = self.model.forward_sync(A_min, a_ctrl, belief_graph)
                            predictions.append(pred)
                            labels.append(sync_label)
                        
                        predictions = torch.cat(predictions)
                        labels = torch.tensor(labels, dtype=torch.float32, device=self.device)
                        
                        loss = criterion(predictions.squeeze(), labels)
                        val_loss += loss.item()
                        n_val_batches += 1
                
                avg_val_loss = val_loss / n_val_batches
                val_losses.append(avg_val_loss)
            
            if (epoch + 1) % 10 == 0:
                val_str = f", Val Loss = {val_losses[-1]:.4f}" if val_losses else ""
                print(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.4f}{val_str}")
        
        return {'train_losses': train_losses, 'val_losses': val_losses}
    
    def train_all(self, train_data: List[Dict], val_data: List[Dict] = None,
                  epochs: int = 50, lr: float = 1e-3, batch_size: int = 32):
        """Train all prediction heads sequentially."""
        print("="*80)
        print("Training all surrogate model heads...")
        print("="*80)
        
        # Train MOCU head (most important)
        mocu_results = self.train_mocu(train_data, val_data, epochs, lr, batch_size)
        print()
        
        # Train ERM head
        erm_results = self.train_erm(train_data, val_data, epochs, lr, batch_size)
        print()
        
        # Train Sync head
        sync_results = self.train_sync(train_data, val_data, epochs, lr, batch_size)
        print()
        
        print("="*80)
        print("Training complete!")
        print("="*80)
        
        return {
            'mocu': mocu_results,
            'erm': erm_results,
            'sync': sync_results
        }


def train_surrogate_model(N: int = 5, K: int = 4, n_train: int = 1000, n_val: int = 200,
                         n_theta_samples: int = 20, epochs: int = 50, lr: float = 1e-3, 
                         batch_size: int = 32, device: str = None, 
                         save_path: str = 'trained_surrogate.pth'):
    """Complete training pipeline for the surrogate model."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"✓ Auto-detected device: {device}")
    print("="*80)
    print("SURROGATE MODEL TRAINING PIPELINE")
    print("="*80)
    print(f"Configuration:")
    print(f"  N = {N}, K = {K}")
    print(f"  Training samples: {n_train}, Validation samples: {n_val}")
    print(f"  MOCU/ERM estimation samples: {n_theta_samples}")
    print(f"  Epochs: {epochs}, Learning rate: {lr}")
    print("="*80)
    print()
    
    # Generate training data
    print("Generating training data...")
    train_generator = SyntheticDataGenerator(
        N=N, K=K, n_samples=n_train, n_theta_samples=n_theta_samples
    )
    train_data = train_generator.generate_dataset(seed=42)
    print()
    
    # Generate validation data
    print("Generating validation data...")
    val_generator = SyntheticDataGenerator(
        N=N, K=K, n_samples=n_val, n_theta_samples=n_theta_samples
    )
    val_data = val_generator.generate_dataset(seed=123)
    print()
    
    # Initialize model
    print("Initializing model...")
    model = MPNNSurrogate(mocu_scale=1.0)
    trainer = SurrogateTrainer(model, device=device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print()
    
    # Train model
    results = trainer.train_all(train_data, val_data, epochs, lr, batch_size)
    
    # Save model
    print(f"Saving model to {save_path}...")
    torch.save(model.state_dict(), save_path)
    print(" Model saved successfully!")
    print()
    
    return model, results


if __name__ == "__main__":
    # Train a small model for testing
    print("Testing fixed surrogate training...")
    model, results = train_surrogate_model(
        N=5, K=4, n_train=100, n_val=20, n_theta_samples=10, 
        epochs=20, device=None, save_path='test_surrogate.pth'
    )
    print("\n Training completed successfully!")
    print(f"Final MOCU train loss: {results['mocu']['train_losses'][-1]:.4f}")
    print(f"Final ERM train loss: {results['erm']['train_losses'][-1]:.4f}")
    print(f"Final Sync train loss: {results['sync']['train_losses'][-1]:.4f}")