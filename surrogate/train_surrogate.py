"""
Fixed training pipeline for the MPNN surrogate model.
Uses cached data with proper normalization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple
from surrogate.mpnn_surrogate import MPNNSurrogate

# Try to import tqdm
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class DataCache:
    """Load cached datasets."""
    
    def __init__(self, cache_dir: str = "dataset"):
        self.cache_dir = cache_dir
    
    def get_cache_path(self, split: str, N: int, K: int, n_samples: int, 
                       n_theta_samples: int, seed: int) -> str:
        """Generate cache filename."""
        filename = f"{split}_N{N}_K{K}_n{n_samples}_mc{n_theta_samples}_seed{seed}.pkl"
        return os.path.join(self.cache_dir, filename)
    
    def exists(self, split: str, N: int, K: int, n_samples: int, 
              n_theta_samples: int, seed: int) -> bool:
        """Check if cached data exists."""
        path = self.get_cache_path(split, N, K, n_samples, n_theta_samples, seed)
        return os.path.exists(path)
    
    def load(self, split: str, N: int, K: int, n_samples: int, 
            n_theta_samples: int, seed: int) -> List[Dict]:
        """Load dataset from cache."""
        path = self.get_cache_path(split, N, K, n_samples, n_theta_samples, seed)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Cached data not found: {path}\n"
                f"Run: python generate_data.py --config <config> --split {split}"
            )
        
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
        print(f" Loaded {len(dataset)} samples from cache: {path}")
        return dataset


def normalize_dataset(dataset, stats=None):
    """
    Normalize MOCU labels to mean=0, std=1.
    
    Following Chen et al. (2023) methodology for better training convergence.
    
    Args:
        dataset: List of training samples
        stats: Optional tuple (mean, std) to use. If None, compute from dataset.
    
    Returns:
        normalized_dataset, (mean, std)
    """
    # Collect all MOCU values
    all_mocu = []
    for sample in dataset:
        for step_data in sample['experiment_data']:
            all_mocu.append(step_data['mocu'])
        all_mocu.append(sample['final_mocu'])
    
    # Compute or use provided statistics
    if stats is None:
        mean = float(np.mean(all_mocu))
        std = float(np.std(all_mocu))
        if std < 1e-6:  # Avoid division by zero
            std = 1.0
            print("  �  WARNING: MOCU has near-zero variance!")
    else:
        mean, std = stats
    
    print(f"  Normalizing MOCU: mean={mean:.4f}, std={std:.4f}")
    print(f"  Original range: [{np.min(all_mocu):.4f}, {np.max(all_mocu):.4f}]")
    
    # Normalize all MOCU labels
    for sample in dataset:
        for step_data in sample['experiment_data']:
            step_data['mocu'] = (step_data['mocu'] - mean) / std
        sample['final_mocu'] = (sample['final_mocu'] - mean) / std
    
    # Verify normalization
    all_normalized = []
    for sample in dataset:
        for step_data in sample['experiment_data']:
            all_normalized.append(step_data['mocu'])
        all_normalized.append(sample['final_mocu'])
    
    print(f"  Normalized range: [{np.min(all_normalized):.4f}, {np.max(all_normalized):.4f}]")
    print(f"  Normalized mean: {np.mean(all_normalized):.4f} (should be ~0)")
    print(f"  Normalized std: {np.std(all_normalized):.4f} (should be ~1)")
    
    return dataset, (mean, std)


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
            omega = exp_data['omega']
            
            for a_ctrl, sync_label in exp_data['sync_scores'].items():
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
        
        # Progress bar for epochs
        epoch_iter = tqdm(range(epochs), desc="MOCU Training", ncols=100) if TQDM_AVAILABLE else range(epochs)
        
        for epoch in epoch_iter:
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
            
            # Update progress bar
            if TQDM_AVAILABLE:
                desc = f"MOCU Training - Loss: {avg_train_loss:.4f}"
                if val_losses:
                    desc += f" | Val: {val_losses[-1]:.4f}"
                epoch_iter.set_description(desc)
            elif (epoch + 1) % 10 == 0:
                val_str = f", Val Loss = {val_losses[-1]:.4f}" if val_losses else ""
                print(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.4f}{val_str}")
        
        if TQDM_AVAILABLE:
            epoch_iter.close()
        
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
        
        mocu_results = self.train_mocu(train_data, val_data, epochs, lr, batch_size)
        print()
        
        erm_results = self.train_erm(train_data, val_data, epochs, lr, batch_size)
        print()
        
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
                         save_path: str = 'trained_surrogate.pth',
                         use_cache: bool = True, hidden: int = 64, dropout: float = 0.1,
                         mocu_scale: float = 1.0):
    """Complete training pipeline with normalization - uses cached data if available."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Auto-detected device: {device}")
    
    print("="*80)
    print("SURROGATE MODEL TRAINING PIPELINE")
    print("="*80)
    print(f"Configuration:")
    print(f"  N = {N}, K = {K}")
    print(f"  Training samples: {n_train}, Validation samples: {n_val}")
    print(f"  MOCU/ERM estimation samples: {n_theta_samples}")
    print(f"  Epochs: {epochs}, Learning rate: {lr}")
    print(f"  Use cached data: {use_cache}")
    print("="*80)
    print()
    
    # Load data
    cache = DataCache()
    
    if use_cache:
        # Try to load from cache
        try:
            print("Loading training data from cache...")
            train_data = cache.load("train", N, K, n_train, n_theta_samples, 42)
            print()
            
            print("Loading validation data from cache...")
            val_data = cache.load("val", N, K, n_val, n_theta_samples, 123)
            print()
        except FileNotFoundError as e:
            print(f"\nL Error: {e}")
            print("\nPlease generate data first:")
            print("  python generate_data.py --split both --parallel")
            raise
    else:
        raise NotImplementedError("On-the-fly data generation not implemented")
    
    # ============================================================
    # CRITICAL FIX: Add data normalization here
    # ============================================================
    print("\n" + "="*80)
    print("NORMALIZING MOCU LABELS (Chen et al. 2023)")
    print("="*80)
    
    # Normalize training data and get statistics
    print("Normalizing training data...")
    train_data, (mocu_mean, mocu_std) = normalize_dataset(train_data, stats=None)
    
    # Normalize validation data using training statistics
    print("\nNormalizing validation data (using training stats)...")
    val_data, _ = normalize_dataset(val_data, stats=(mocu_mean, mocu_std))
    
    # Save normalization parameters alongside the model
    norm_path = os.path.join(os.path.dirname(save_path), 'mocu_normalization.pkl')
    with open(norm_path, 'wb') as f:
        pickle.dump({'mean': mocu_mean, 'std': mocu_std}, f)
    print(f"\n Saved normalization parameters to {norm_path}")
    print("="*80 + "\n")
    
    # Initialize model
    print("Initializing model...")
    model = MPNNSurrogate(
        mocu_scale=mocu_scale,
        hidden=hidden,
        dropout=dropout
    )
    
    # Store normalization in model for inference
    model.mocu_mean = mocu_mean
    model.mocu_std = mocu_std
    
    trainer = SurrogateTrainer(model, device=device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print()
    
    # Train model
    results = trainer.train_all(train_data, val_data, epochs, lr, batch_size)
    
    # Save model
    if save_path:
        print(f"Saving model to {save_path}...")
        torch.save(model.state_dict(), save_path)
        print(" Model saved successfully!")
        print()
    
    return model, results


def load_surrogate_with_normalization(model_path, hidden=64, dropout=0.1, mocu_scale=1.0, device='cpu'):
    """
    Load surrogate model and its normalization parameters.
    
    The model will automatically denormalize predictions during inference.
    """
    # Load model
    model = MPNNSurrogate(
        mocu_scale=mocu_scale,
        hidden=hidden,
        dropout=dropout
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    # Load normalization parameters
    norm_path = os.path.join(os.path.dirname(model_path), 'mocu_normalization.pkl')
    if os.path.exists(norm_path):
        with open(norm_path, 'rb') as f:
            norm_params = pickle.load(f)
        model.mocu_mean = norm_params['mean']
        model.mocu_std = norm_params['std']
        print(f" Loaded normalization: mean={model.mocu_mean:.4f}, std={model.mocu_std:.4f}")
    else:
        print("�  Warning: No normalization file found. Model may not predict correctly.")
        model.mocu_mean = 0.0
        model.mocu_std = 1.0
    
    model.to(device)
    model.eval()
    
    return model


if __name__ == "__main__":
    print("Testing surrogate training with cached data...")
    model, results = train_surrogate_model(
        N=5, K=4, n_train=100, n_val=20, n_theta_samples=10, 
        epochs=20, device=None, save_path='test_surrogate.pth',
        use_cache=True
    )
    print("\n Training completed successfully!")
    print(f"Final MOCU train loss: {results['mocu']['train_losses'][-1]:.4f}")
    print(f"Final ERM train loss: {results['erm']['train_losses'][-1]:.4f}")
    print(f"Final Sync train loss: {results['sync']['train_losses'][-1]:.4f}")