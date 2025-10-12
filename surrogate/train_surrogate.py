"""
Training pipeline for the MPNN surrogate model.
Trains the model to predict MOCU, ERM, and Sync probabilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple
import json
import os
from .mpnn_surrogate import MPNNSurrogate
from data_generation.synthetic_data import SyntheticDataGenerator


class SurrogateDataset(Dataset):
    """Dataset for training the MPNN surrogate model."""
    
    def __init__(self, data: List[Dict], task: str = 'mocu'):
        """
        Args:
            data: List of experiment data from SyntheticDataGenerator
            task: 'mocu', 'erm', or 'sync'
        """
        self.data = data
        self.task = task
        self.samples = self._prepare_samples()
    
    def _prepare_samples(self) -> List[Dict]:
        """Prepare samples for the specific task."""
        samples = []
        
        for exp_data in self.data:
            if self.task == 'mocu':
                # Use final belief graph and MOCU label
                samples.append({
                    'belief_graph': exp_data['final_belief_graph'],
                    'label': exp_data['labels']['mocu']
                })
            
            elif self.task == 'erm':
                # Use belief graphs from each step and ERM labels
                for step_data in exp_data['experiment_data']:
                    for xi, erm_score in exp_data['labels']['erm_scores'].items():
                        samples.append({
                            'belief_graph': step_data['belief_graph'],
                            'candidate_pair': xi,
                            'label': erm_score
                        })
            
            elif self.task == 'sync':
                # Use final belief graph and sync labels
                for a_ctrl, sync_prob in exp_data['labels']['sync_scores'].items():
                    samples.append({
                        'belief_graph': exp_data['final_belief_graph'],
                        'a_ctrl': a_ctrl,
                        'label': 1.0 if sync_prob else 0.0
                    })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class SurrogateTrainer:
    """Trainer for the MPNN surrogate model."""
    
    def __init__(self, model: MPNNSurrogate, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        
    def train_mocu(self, train_data: List[Dict], val_data: List[Dict] = None,
                   epochs: int = 50, lr: float = 1e-3, batch_size: int = 32):
        """Train MOCU prediction head."""
        print("Training MOCU prediction head...")
        
        train_dataset = SurrogateDataset(train_data, task='mocu')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if val_data:
            val_dataset = SurrogateDataset(val_data, task='mocu')
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                belief_graphs = batch['belief_graph']
                labels = torch.tensor(batch['label'], dtype=torch.float, device=self.device)
                
                predictions = []
                for bg in belief_graphs:
                    pred = self.model.forward_mocu(bg)
                    predictions.append(pred)
                
                predictions = torch.cat(predictions)
                loss = criterion(predictions.squeeze(), labels)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        belief_graphs = batch['belief_graph']
                        labels = torch.tensor(batch['label'], dtype=torch.float, device=self.device)
                        
                        predictions = []
                        for bg in belief_graphs:
                            pred = self.model.forward_mocu(bg)
                            predictions.append(pred)
                        
                        predictions = torch.cat(predictions)
                        loss = criterion(predictions.squeeze(), labels)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                val_losses.append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.4f}, "
                      f"Val Loss = {val_losses[-1] if val_losses else 'N/A':.4f}")
        
        return {'train_losses': train_losses, 'val_losses': val_losses}
    
    def train_erm(self, train_data: List[Dict], val_data: List[Dict] = None,
                  epochs: int = 50, lr: float = 1e-3, batch_size: int = 32):
        """Train ERM prediction head."""
        print("Training ERM prediction head...")
        
        train_dataset = SurrogateDataset(train_data, task='erm')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if val_data:
            val_dataset = SurrogateDataset(val_data, task='erm')
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                
                belief_graphs = batch['belief_graph']
                candidate_pairs = batch['candidate_pair']
                labels = torch.tensor(batch['label'], dtype=torch.float, device=self.device)
                
                predictions = []
                for bg, xi in zip(belief_graphs, candidate_pairs):
                    pred = self.model.forward_erm(bg, xi)
                    predictions.append(pred)
                
                predictions = torch.cat(predictions)
                loss = criterion(predictions.squeeze(), labels)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        belief_graphs = batch['belief_graph']
                        candidate_pairs = batch['candidate_pair']
                        labels = torch.tensor(batch['label'], dtype=torch.float, device=self.device)
                        
                        predictions = []
                        for bg, xi in zip(belief_graphs, candidate_pairs):
                            pred = self.model.forward_erm(bg, xi)
                            predictions.append(pred)
                        
                        predictions = torch.cat(predictions)
                        loss = criterion(predictions.squeeze(), labels)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                val_losses.append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.4f}, "
                      f"Val Loss = {val_losses[-1] if val_losses else 'N/A':.4f}")
        
        return {'train_losses': train_losses, 'val_losses': val_losses}
    
    def train_sync(self, train_data: List[Dict], val_data: List[Dict] = None,
                   epochs: int = 50, lr: float = 1e-3, batch_size: int = 32):
        """Train Sync prediction head."""
        print("Training Sync prediction head...")
        
        train_dataset = SurrogateDataset(train_data, task='sync')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if val_data:
            val_dataset = SurrogateDataset(val_data, task='sync')
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                
                belief_graphs = batch['belief_graph']
                a_ctrl_values = batch['a_ctrl']
                labels = torch.tensor(batch['label'], dtype=torch.float, device=self.device)
                
                predictions = []
                for bg, a_ctrl in zip(belief_graphs, a_ctrl_values):
                    pred = self.model.forward_sync(bg, a_ctrl, bg)  # Using belief_graph as A_min proxy
                    predictions.append(pred)
                
                predictions = torch.cat(predictions)
                loss = criterion(predictions.squeeze(), labels)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        belief_graphs = batch['belief_graph']
                        a_ctrl_values = batch['a_ctrl']
                        labels = torch.tensor(batch['label'], dtype=torch.float, device=self.device)
                        
                        predictions = []
                        for bg, a_ctrl in zip(belief_graphs, a_ctrl_values):
                            pred = self.model.forward_sync(bg, a_ctrl, bg)
                            predictions.append(pred)
                        
                        predictions = torch.cat(predictions)
                        loss = criterion(predictions.squeeze(), labels)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                val_losses.append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.4f}, "
                      f"Val Loss = {val_losses[-1] if val_losses else 'N/A':.4f}")
        
        return {'train_losses': train_losses, 'val_losses': val_losses}
    
    def train_all(self, train_data: List[Dict], val_data: List[Dict] = None,
                  epochs: int = 50, lr: float = 1e-3, batch_size: int = 32):
        """Train all prediction heads."""
        print("Training all surrogate model heads...")
        
        # Train each head
        mocu_results = self.train_mocu(train_data, val_data, epochs, lr, batch_size)
        erm_results = self.train_erm(train_data, val_data, epochs, lr, batch_size)
        sync_results = self.train_sync(train_data, val_data, epochs, lr, batch_size)
        
        return {
            'mocu': mocu_results,
            'erm': erm_results,
            'sync': sync_results
        }


def train_surrogate_model(N: int = 5, K: int = 4, n_train: int = 1000, n_val: int = 200,
                         epochs: int = 50, lr: float = 1e-3, batch_size: int = 32,
                         device: str = 'cpu', save_path: str = 'surrogate_model.pth'):
    """Complete training pipeline for the surrogate model."""
    
    print("Generating training data...")
    train_generator = SyntheticDataGenerator(N=N, K=K, n_samples=n_train)
    train_data = train_generator.generate_dataset(seed=42)
    
    print("Generating validation data...")
    val_generator = SyntheticDataGenerator(N=N, K=K, n_samples=n_val)
    val_data = val_generator.generate_dataset(seed=123)
    
    print("Initializing model...")
    model = MPNNSurrogate(mocu_scale=1.0)
    trainer = SurrogateTrainer(model, device=device)
    
    print("Training model...")
    results = trainer.train_all(train_data, val_data, epochs, lr, batch_size)
    
    print("Saving model...")
    torch.save(model.state_dict(), save_path)
    
    print("Training complete!")
    return model, results


if __name__ == "__main__":
    # Train a small model for testing
    model, results = train_surrogate_model(
        N=5, K=4, n_train=500, n_val=100, epochs=20, 
        device='cpu', save_path='test_surrogate.pth'
    )
    print("Training completed successfully!")
