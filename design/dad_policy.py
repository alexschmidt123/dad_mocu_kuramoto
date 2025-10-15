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
        
        nN = node.shape[0]
        nE = edge.shape[0]
        
        hn = self.node(node).mean(0, keepdim=True)
        
        if nE > 0:
            he = self.edge(edge).mean(0, keepdim=True)
        else:
            he = torch.zeros((1, self.hidden), device=device)
        
        return torch.cat([hn, he], dim=-1)


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
        
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        out, (h, c) = self.lstm(x)
        h = h.squeeze(0)
        
        return h


class DADPolicy(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        
        self.encoder = SimpleEncoder(hidden=hidden)
        self.hist = LSTMHistory(in_dim=3, hidden=hidden)
        
        # Fusion: graph (2*hidden) + history (hidden) + candidate (6)
        fusion_input_dim = 3 * hidden + 6
        
        self.fuse = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, belief_graph, cand_pairs, hist_tokens, N):
        z_graph = self.encoder(belief_graph)
        z_hist = self.hist(hist_tokens)
        
        scores = []
        device = next(self.parameters()).device
        
        # Get node features for candidate encoding
        node_feats = torch.as_tensor(
            belief_graph["node_feats"], 
            dtype=torch.float, 
            device=device
        )
        
        for (i, j) in cand_pairs:
            node_i = node_feats[i]
            node_j = node_feats[j]
            
            cand = torch.cat([
                node_i,
                node_j,
                torch.tensor([i / (N - 1 + 1e-9), j / (N - 1 + 1e-9)], 
                           dtype=torch.float, device=device)
            ]).unsqueeze(0)
            
            z = torch.cat([z_graph, z_hist, cand], dim=-1)
            score = self.fuse(z).squeeze(1)
            scores.append(score)
        
        scores = torch.cat(scores, dim=0)
        
        # NO score normalization - keep raw scores for RL training
        
        return scores

    def choose(self, env, hist_tokens):
        cands = env.candidate_pairs()
        g = env.features()
        scores = self.forward(g, cands, hist_tokens, env.N).detach().cpu().numpy()
        return cands[int(scores.argmin())]