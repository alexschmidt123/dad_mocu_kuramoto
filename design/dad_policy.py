import torch
import torch.nn as nn

class SimpleEncoder(nn.Module):
    def __init__(self, node_dim=2, edge_dim=5, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.node = nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU())
        self.edge = nn.Sequential(nn.Linear(edge_dim, hidden), nn.ReLU())
    
    def forward(self, belief_graph):
        device = next(self.parameters()).device if list(self.parameters()) else 'cpu'
        
        node = torch.as_tensor(belief_graph["node_feats"], dtype=torch.float, device=device)
        edge = torch.as_tensor(belief_graph["edge_feats"], dtype=torch.float, device=device)
        
        hn = self.node(node).mean(0, keepdim=True)  # (1, hidden)
        
        if edge.numel() > 0:
            he = self.edge(edge).mean(0, keepdim=True)  # (1, hidden)
        else:
            he = torch.zeros((1, self.hidden), device=device)
        
        return torch.cat([hn, he], dim=-1)  # (1, 2*hidden)


class LSTMHistory(nn.Module):
    def __init__(self, in_dim=3, hidden=64):
        super().__init__()
        self.hidden_size = hidden
        self.lstm = nn.LSTM(in_dim, hidden, 1, batch_first=True)
    
    def forward(self, hist_tokens):
        device = next(self.parameters()).device
        
        if not hist_tokens:
            return torch.zeros((1, self.hidden_size), device=device)
        
        x = torch.as_tensor(hist_tokens, dtype=torch.float, device=device)
        
        # Ensure x is 3D: (batch=1, seq_len, in_dim=3)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (seq_len, 3) -> (1, seq_len, 3)
        
        out, (h, c) = self.lstm(x)
        # h has shape: (num_layers=1, batch=1, hidden)
        # We want: (batch=1, hidden) for concatenation
        # So remove the num_layers dimension
        return h.squeeze(0)  # (1, 1, hidden) -> (1, hidden)


class DADPolicy(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        
        self.encoder = SimpleEncoder(hidden=hidden)
        self.hist = LSTMHistory(in_dim=3, hidden=hidden)
        
        # Fusion layer: 2*hidden (encoder) + hidden (history) + 2 (candidate) = 3*hidden + 2
        fusion_input_dim = 3 * hidden + 2
        self.fuse = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, belief_graph, cand_pairs, hist_tokens, N):
        z_graph = self.encoder(belief_graph)  # (1, 2*hidden)
        z_hist = self.hist(hist_tokens)        # (1, hidden)
        
        scores = []
        device = next(self.parameters()).device
        
        for (i, j) in cand_pairs:
            cand = torch.tensor([[i / (N - 1 + 1e-9), j / (N - 1 + 1e-9)]], 
                               dtype=torch.float, device=device)  # (1, 2)
            
            # All tensors now have shape (1, D), can concatenate along dim=-1
            z = torch.cat([z_graph, z_hist, cand], dim=-1)  # (1, 3*hidden + 2)
            score = self.fuse(z).squeeze(1)  # (1,)
            scores.append(score)
        
        return torch.cat(scores, dim=0)  # (num_candidates,)

    def choose(self, env, hist_tokens):
        cands = env.candidate_pairs()
        g = env.features()
        scores = self.forward(g, cands, hist_tokens, env.N).detach().cpu().numpy()
        return cands[int(scores.argmin())]


if __name__ == "__main__":
    print("Testing DAD Policy dimensions...")
    
    # Create dummy inputs
    N = 5
    belief_graph = {
        'node_feats': torch.randn(N, 2).numpy(),
        'edge_feats': torch.randn(10, 5).numpy(),
        'edge_index': torch.randint(0, N, (2, 10)).numpy()
    }
    
    cand_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Test with different history lengths
    test_histories = [
        [],  # Empty history
        [[0.2, 0.4, 1.0]],  # One experiment
        [[0.2, 0.4, 1.0], [0.1, 0.3, 0.0]],  # Two experiments
        [[0.2, 0.4, 1.0], [0.1, 0.3, 0.0], [0.3, 0.5, 1.0]],  # Three
    ]
    
    policy = DADPolicy(hidden=64)
    
    for i, hist_tokens in enumerate(test_histories):
        print(f"\nTest {i+1}: {len(hist_tokens)} experiments in history")
        try:
            scores = policy.forward(belief_graph, cand_pairs, hist_tokens, N)
            print(f"  SUCCESS: scores shape = {scores.shape}")
            print(f"  Sample scores: {scores[:3].detach().numpy()}")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nAll tests completed!")