import torch, torch.nn as nn

class SimpleEncoder(nn.Module):
    def __init__(self, node_dim=2, edge_dim=5, hidden=64):
        super().__init__()
        self.node = nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU())
        self.edge = nn.Sequential(nn.Linear(edge_dim, hidden), nn.ReLU())
    def forward(self, belief_graph):
        node = torch.as_tensor(belief_graph["node_feats"], dtype=torch.float)
        edge = torch.as_tensor(belief_graph["edge_feats"], dtype=torch.float)
        hn = self.node(node).mean(0, keepdim=True)
        he = self.edge(edge).mean(0, keepdim=True) if edge.numel()>0 else torch.zeros_like(hn)
        return torch.cat([hn, he], dim=-1)

class LSTMHistory(nn.Module):
    def __init__(self, in_dim=3, hidden=64):
        super().__init__(); self.lstm = nn.LSTM(in_dim, hidden, 1, batch_first=True)
    def forward(self, hist_tokens):
        if not hist_tokens: return torch.zeros((1, self.lstm.hidden_size))
        x = torch.as_tensor(hist_tokens, dtype=torch.float).unsqueeze(0)
        out, (h, c) = self.lstm(x); return h[-1].unsqueeze(0)

class DADPolicy(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.encoder = SimpleEncoder(hidden=hidden)
        self.hist = LSTMHistory(in_dim=3, hidden=hidden)
        self.fuse = nn.Sequential(nn.Linear(2*hidden+2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, belief_graph, cand_pairs, hist_tokens, N):
        z_graph = self.encoder(belief_graph); z_hist = self.hist(hist_tokens)
        scores = []
        for (i,j) in cand_pairs:
            cand = torch.tensor([[i/(N-1+1e-9), j/(N-1+1e-9)]], dtype=torch.float)
            z = torch.cat([z_graph, z_hist, cand], dim=-1)
            scores.append(self.fuse(z).squeeze(1))
        return torch.cat(scores, dim=0).squeeze(1)

    def choose(self, env, hist_tokens):
        cands = env.candidate_pairs(); g = env.features()
        scores = self.forward(g, cands, hist_tokens, env.N).detach().cpu().numpy()
        return cands[int(scores.argmin())]
