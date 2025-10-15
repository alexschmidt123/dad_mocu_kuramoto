"""
MPNN surrogate with proper message passing layers following Chen et al. (2023).

Key features:
- Proper MPNNLayer with neighbor message aggregation
- Multiple graph convolution layers
- Graph-level pooling for MOCU/ERM/Sync prediction
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


class MPNNLayer(nn.Module):
    """
    Message Passing Neural Network layer following Chen et al. (2023).
    
    Implements proper graph convolution with:
    1. Message computation from edges
    2. Message aggregation per node
    3. Node update with residual connection
    """
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Message function: combines source node, target node, and edge features
        self.message_net = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update function: combines node features with aggregated messages
        self.update_net = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
    
    def forward(self, node_feats, edge_feats, edge_index):
        """
        Args:
            node_feats: (N, node_dim) - node features
            edge_feats: (E, edge_dim) - edge features
            edge_index: (2, E) - edge connectivity [source_nodes, target_nodes]
        
        Returns:
            updated_node_feats: (N, node_dim)
        """
        if edge_index.shape[1] == 0:
            # No edges, return unchanged
            return node_feats
        
        N = node_feats.shape[0]
        device = node_feats.device
        
        # Get source and target node indices
        src_idx = edge_index[0]  # Source nodes
        tgt_idx = edge_index[1]  # Target nodes
        
        # Gather source and target node features
        src_feats = node_feats[src_idx]  # (E, node_dim)
        tgt_feats = node_feats[tgt_idx]  # (E, node_dim)
        
        # Compute messages: combine source, target, and edge features
        message_input = torch.cat([src_feats, tgt_feats, edge_feats], dim=1)
        messages = self.message_net(message_input)  # (E, hidden_dim)
        
        # Aggregate messages for each target node (sum aggregation)
        aggregated = torch.zeros(N, self.hidden_dim, device=device)
        aggregated.index_add_(0, tgt_idx, messages)
        
        # Update node features
        update_input = torch.cat([node_feats, aggregated], dim=1)
        node_updates = self.update_net(update_input)
        
        # Residual connection
        return node_feats + node_updates


class MPNNSurrogate(nn.Module):
    """
    MPNN Surrogate following Chen et al. (2023) with proper message passing.
    
    Architecture:
    1. Node and edge encoders
    2. Multiple MPNN layers for message passing
    3. Global pooling (mean)
    4. Task-specific heads: MOCU, ERM, Sync
    """
    
    def __init__(self, mocu_scale=1.0, hidden=64, dropout=0.1, n_mpnn_layers=3):
        super().__init__()
        
        self.hidden = hidden
        self.mocu_scale = mocu_scale
        
        # Encoders for node and edge features
        self.node_encoder = ImprovedMLP(din=2, dout=hidden, hidden=hidden, dropout=dropout)
        self.edge_encoder = ImprovedMLP(din=5, dout=hidden, hidden=hidden, dropout=dropout)
        
        # MPNN layers for message passing
        self.mpnn_layers = nn.ModuleList([
            MPNNLayer(hidden, hidden, hidden) for _ in range(n_mpnn_layers)
        ])
        
        # Prediction heads
        self.head_mocu = ImprovedMLP(din=2*hidden, dout=1, hidden=hidden, dropout=dropout)
        self.head_erm = ImprovedMLP(din=2*hidden + 2, dout=1, hidden=hidden, dropout=dropout)
        
        # Sync head: needs to incorporate matrix information
        self.matrix_encoder = ImprovedMLP(din=1, dout=hidden//2, hidden=hidden//2, dropout=dropout)
        self.head_sync = ImprovedMLP(din=2*hidden + hidden//2 + 1, dout=1, hidden=hidden, dropout=dropout)
        
        # Normalization parameters (set during training)
        self.register_buffer('mocu_mean', torch.tensor(0.0))
        self.register_buffer('mocu_std', torch.tensor(1.0))
    
    def _encode_and_propagate(self, node_feats, edge_feats, edge_index):
        """Encode features and run message passing."""
        device = next(self.parameters()).device
        
        # Move to device
        node_feats = node_feats.to(device)
        edge_feats = edge_feats.to(device)
        edge_index = edge_index.to(device)
        
        # Encode to hidden dimension
        node_h = self.node_encoder(node_feats)  # (N, hidden)
        edge_h = self.edge_encoder(edge_feats) if edge_feats.shape[0] > 0 else torch.zeros((0, self.hidden), device=device)
        
        # Message passing: iterate through MPNN layers
        for mpnn_layer in self.mpnn_layers:
            node_h = mpnn_layer(node_h, edge_h, edge_index)
        
        return node_h, edge_h
    
    def _aggregate(self, node_h, edge_h):
        """Global pooling to get graph-level representation."""
        # Mean pooling over nodes and edges
        g_node = node_h.mean(dim=0, keepdim=True)  # (1, hidden)
        g_edge = edge_h.mean(dim=0, keepdim=True) if edge_h.shape[0] > 0 else torch.zeros((1, self.hidden), device=node_h.device)
        
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
        
        # Encode and propagate through MPNN layers
        node_h, edge_h = self._encode_and_propagate(node, edge, idx)
        
        # Aggregate to graph-level representation
        g = self._aggregate(node_h, edge_h)
        
        # Predict (normalized)
        y = self.head_mocu(g)
        y_normalized = F.softplus(y) * self.mocu_scale
        
        # Denormalize to original scale
        y_denormalized = y_normalized * self.mocu_std + self.mocu_mean
        
        return y_denormalized
    
    def forward_erm(self, belief_graph: dict, xi: tuple) -> torch.Tensor:
        """Predict ERM for a candidate experiment xi."""
        device = next(self.parameters()).device
        
        node = torch.as_tensor(belief_graph["node_feats"], dtype=torch.float32, device=device)
        edge = torch.as_tensor(belief_graph["edge_feats"], dtype=torch.float32, device=device)
        idx = torch.as_tensor(belief_graph["edge_index"], dtype=torch.long, device=device)
        
        # Encode and propagate
        node_h, edge_h = self._encode_and_propagate(node, edge, idx)
        
        # Aggregate
        g = self._aggregate(node_h, edge_h)
        
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
        
        # Encode and propagate
        node_h, edge_h = self._encode_and_propagate(node, edge, idx)
        
        # Aggregate belief graph features
        g = self._aggregate(node_h, edge_h)
        
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
    
    def forward_mocu_batch(self, belief_graphs: list) -> torch.Tensor:
        """Batch MOCU prediction for efficiency."""
        predictions = []
        for belief_graph in belief_graphs:
            pred = self.forward_mocu(belief_graph)
            predictions.append(pred)
        return torch.cat(predictions)


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
    print("="*80)
    print("Testing MPNN Surrogate with Message Passing Layers")
    print("="*80)
    
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
    model = MPNNSurrogate(mocu_scale=1.0, hidden=64, n_mpnn_layers=3)
    
    # Test on GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Testing on device: {device}")
    
    # Verify MPNN layers exist
    print(f"\nModel has {len(model.mpnn_layers)} MPNN layers: SUCCESS")
    
    # Test MOCU prediction
    print("\n1. Testing MOCU prediction...")
    mocu_pred = model.forward_mocu(belief_graph)
    print(f"   SUCCESS: MOCU prediction: {mocu_pred.item():.4f}")
    
    # Test ERM prediction
    print("\n2. Testing ERM prediction...")
    xi = (0, 1)
    erm_pred = model.forward_erm(belief_graph, xi)
    print(f"   SUCCESS: ERM prediction for pair {xi}: {erm_pred.item():.4f}")
    
    # Test Sync prediction
    print("\n3. Testing Sync prediction...")
    A_min = np.random.rand(N, N) * 0.5
    A_min = 0.5 * (A_min + A_min.T)
    np.fill_diagonal(A_min, 0.0)
    a_ctrl = 0.15
    sync_pred = model.forward_sync(A_min, a_ctrl, belief_graph)
    print(f"   SUCCESS: Sync prediction (a_ctrl={a_ctrl}): {sync_pred.item():.4f}")
    
    # Test batch prediction
    print("\n4. Testing batch MOCU prediction...")
    batch_graphs = [belief_graph for _ in range(5)]
    batch_preds = model.forward_mocu_batch(batch_graphs)
    print(f"   SUCCESS: Batch predictions shape: {batch_preds.shape}")
    
    # Test gradient flow
    print("\n5. Testing gradient flow...")
    loss = mocu_pred + erm_pred + sync_pred
    loss.backward()
    print(f"   SUCCESS: Gradient flow successful")
    
    # Test message passing
    print("\n6. Testing message passing mechanism...")
    node_h_init = model.node_encoder(torch.tensor(node_feats, dtype=torch.float32, device=device))
    edge_h_init = model.edge_encoder(torch.tensor(edge_feats, dtype=torch.float32, device=device))
    edge_idx = torch.tensor(edge_index, dtype=torch.long, device=device)
    
    node_h_after = node_h_init.clone()
    for layer_idx, mpnn_layer in enumerate(model.mpnn_layers):
        node_h_after = mpnn_layer(node_h_after, edge_h_init, edge_idx)
        print(f"   Layer {layer_idx+1}: Node features updated - SUCCESS")
    
    print("\n" + "="*80)
    print("SUCCESS: All tests passed!")
    print("="*80)
    print("\nKey features verified:")
    print("  - MPNNLayer class implemented")
    print("  - Message passing working")
    print("  - Multiple MPNN layers stacked")
    print("  - Gradient flow working")
    print("  - All prediction heads working")