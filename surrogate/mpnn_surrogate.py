"""
Improved MPNN surrogate model with proper device handling.
Fixed sync head to properly use A_min information.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedMLP(nn.Module):
    """Improved MLP with layer normalization and dropout."""
    def __init__(self, din, dout, hidden=64, n_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = din
        
        for i in range(n_layers - 1):
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden
        
        layers.append(nn.Linear(prev_dim, dout))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class MPNNSurrogate(nn.Module):
    """
    Improved MPNN surrogate for Kuramoto experiment design.
    
    Predicts:
    - MOCU: Model Optimality Criterion Under Uncertainty
    - ERM: Expected Remaining MOCU for candidate experiments
    - Sync: Synchronization probability given A_min and a_ctrl
    """
    
    def __init__(self, mocu_scale=1.0, hidden=64, dropout=0.1):
        super().__init__()
        
        # Encoders for node and edge features
        self.node_encoder = ImprovedMLP(din=2, dout=hidden, hidden=hidden, dropout=dropout)
        self.edge_encoder = ImprovedMLP(din=5, dout=hidden, hidden=hidden, dropout=dropout)
        
        # Prediction heads
        self.head_mocu = ImprovedMLP(din=2*hidden, dout=1, hidden=hidden, dropout=dropout)
        self.head_erm = ImprovedMLP(din=2*hidden + 2, dout=1, hidden=hidden, dropout=dropout)
        
        # Sync head: needs to incorporate matrix information
        self.matrix_encoder = ImprovedMLP(din=1, dout=hidden//2, hidden=hidden//2, dropout=dropout)
        self.head_sync = ImprovedMLP(din=2*hidden + hidden//2 + 1, dout=1, hidden=hidden, dropout=dropout)
        
        self.mocu_scale = mocu_scale
        self.hidden = hidden
    
    def _aggregate(self, node_feats, edge_feats, edge_index):
        """Aggregate node and edge features into graph-level representation."""
        device = next(self.parameters()).device  # Get model device
        
        # Move inputs to model device
        node_feats = node_feats.to(device)
        edge_feats = edge_feats.to(device)
        
        nN = node_feats.shape[0]
        nE = edge_feats.shape[0]
        
        # Encode features
        hn = self.node_encoder(node_feats)  # (N, hidden)
        he = self.edge_encoder(edge_feats) if nE > 0 else torch.zeros((0, self.hidden), device=device)
        
        # Global pooling (mean)
        g_node = hn.mean(dim=0, keepdim=True)  # (1, hidden)
        g_edge = he.mean(dim=0, keepdim=True) if nE > 0 else torch.zeros((1, self.hidden), device=device)
        
        return torch.cat([g_node, g_edge], dim=-1)  # (1, 2*hidden)


    def forward_mocu(self, belief_graph: dict) -> torch.Tensor:
        """
        Predict MOCU from belief graph with automatic denormalization.
        
        The model is trained on normalized labels (mean=0, std=1).
        This method automatically denormalizes predictions to original scale.
        """
        device = next(self.parameters()).device
        
        node = torch.as_tensor(belief_graph["node_feats"], dtype=torch.float32, device=device)
        edge = torch.as_tensor(belief_graph["edge_feats"], dtype=torch.float32, device=device)
        idx = torch.as_tensor(belief_graph["edge_index"], dtype=torch.long, device=device)
        
        g = self._aggregate(node, edge, idx)
        y = self.head_mocu(g)
        
        # Model outputs normalized predictions
        y_normalized = F.softplus(y) * self.mocu_scale
        
        # Denormalize to original scale if normalization params are available
        if hasattr(self, 'mocu_mean') and hasattr(self, 'mocu_std'):
            # y_original = y_normalized * std + mean
            y_denormalized = y_normalized * self.mocu_std + self.mocu_mean
            return y_denormalized
        else:
            # No normalization params - return as is (backward compatibility)
            return y_normalized
    
    def forward_erm(self, belief_graph: dict, xi: tuple) -> torch.Tensor:
        """Predict ERM for a candidate experiment xi."""
        device = next(self.parameters()).device
        
        node = torch.as_tensor(belief_graph["node_feats"], dtype=torch.float32, device=device)
        edge = torch.as_tensor(belief_graph["edge_feats"], dtype=torch.float32, device=device)
        idx = torch.as_tensor(belief_graph["edge_index"], dtype=torch.long, device=device)
        
        g = self._aggregate(node, edge, idx)
        
        # Encode candidate pair
        N = node.shape[0]
        i, j = xi
        cand = torch.tensor([[float(i) / (N - 1 + 1e-9), float(j) / (N - 1 + 1e-9)]], 
                           dtype=torch.float32, device=device)
        
        z = torch.cat([g, cand], dim=-1)
        return F.softplus(self.head_erm(z))
    
    def forward_sync(self, A_min: np.ndarray, a_ctrl: float, belief_graph: dict) -> torch.Tensor:
        """
        Predict synchronization probability given A_min matrix and control parameter.
        """
        device = next(self.parameters()).device
        
        node = torch.as_tensor(belief_graph["node_feats"], dtype=torch.float32, device=device)
        edge = torch.as_tensor(belief_graph["edge_feats"], dtype=torch.float32, device=device)
        idx = torch.as_tensor(belief_graph["edge_index"], dtype=torch.long, device=device)
        
        # Aggregate belief graph features
        g = self._aggregate(node, edge, idx)
        
        # Encode A_min matrix information
        N = A_min.shape[0]
        a_min_values = []
        for i in range(N):
            for j in range(i+1, N):
                a_min_values.append(A_min[i, j])
        
        if a_min_values:
            a_min_tensor = torch.tensor(a_min_values, dtype=torch.float32, device=device).unsqueeze(1)
            matrix_encoding = self.matrix_encoder(a_min_tensor).mean(dim=0, keepdim=True)
        else:
            matrix_encoding = torch.zeros((1, self.hidden//2), device=device)
        
        # Encode control parameter
        u = torch.tensor([[a_ctrl]], dtype=torch.float32, device=device)
        
        # Concatenate all features
        z = torch.cat([g, matrix_encoding, u], dim=-1)
        
        # Predict sync probability
        return torch.sigmoid(self.head_sync(z))


# For backward compatibility
class TinyMLP(nn.Module):
    """Legacy MLP - use ImprovedMLP for new code."""
    def __init__(self, din, dout, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, dout),
        )
    
    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    # Test the improved model
    print("Testing improved MPNN surrogate...")
    
    # Create dummy belief graph
    N = 5
    node_feats = np.random.randn(N, 2)
    edge_feats = np.random.randn(10, 5)
    edge_index = np.array([[i, j] for i in range(N) for j in range(i+1, N)]).T
    
    belief_graph = {
        'node_feats': node_feats,
        'edge_feats': edge_feats,
        'edge_index': edge_index
    }
    
    # Create model
    model = MPNNSurrogate(mocu_scale=1.0, hidden=64)
    
    # Test on GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Testing on device: {device}")
    
    # Test MOCU prediction
    mocu_pred = model.forward_mocu(belief_graph)
    print(f"✓ MOCU prediction: {mocu_pred.item():.4f}")
    
    # Test ERM prediction
    xi = (0, 1)
    erm_pred = model.forward_erm(belief_graph, xi)
    print(f"✓ ERM prediction for pair {xi}: {erm_pred.item():.4f}")
    
    # Test Sync prediction
    A_min = np.random.rand(N, N) * 0.5
    A_min = 0.5 * (A_min + A_min.T)
    np.fill_diagonal(A_min, 0.0)
    a_ctrl = 0.15
    sync_pred = model.forward_sync(A_min, a_ctrl, belief_graph)
    print(f"✓ Sync prediction (a_ctrl={a_ctrl}): {sync_pred.item():.4f}")
    
    # Test gradient flow
    loss = mocu_pred + erm_pred + sync_pred
    loss.backward()
    print(f"✓ Gradient flow successful")
    
    print("\n✓ All tests passed!")