import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyMLP(nn.Module):
    def __init__(self, din, dout, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, dout),
        )
    def forward(self, x): return self.net(x)

class MPNNSurrogate(nn.Module):
    def __init__(self, mocu_scale=1.0):
        super().__init__()
        self.node_encoder = TinyMLP(din=2, dout=16)
        self.edge_encoder = TinyMLP(din=5, dout=16)
        self.head_mocu = TinyMLP(din=32, dout=1)
        self.head_erm  = TinyMLP(din=34, dout=1)  # +2 for candidate (i,j)
        self.head_sync = TinyMLP(din=33, dout=1)  # +1 for a_ctrl
        self.mocu_scale = mocu_scale

    def _aggregate(self, node_feats, edge_feats, edge_index):
        device = node_feats.device
        nN = node_feats.shape[0]; nE = edge_feats.shape[0]
        hn = self.node_encoder(node_feats)  # (N,16)
        he = self.edge_encoder(edge_feats)  # (E,16)
        g_node = hn.mean(dim=0, keepdim=True)
        g_edge = he.mean(dim=0, keepdim=True) if nE>0 else torch.zeros((1,16), device=device)
        return torch.cat([g_node, g_edge], dim=-1)  # (1,32)

    def forward_mocu(self, belief_graph: dict) -> torch.Tensor:
        node = torch.as_tensor(belief_graph["node_feats"], dtype=torch.float)
        edge = torch.as_tensor(belief_graph["edge_feats"], dtype=torch.float)
        idx  = torch.as_tensor(belief_graph["edge_index"], dtype=torch.long)
        g = self._aggregate(node, edge, idx)
        y = self.head_mocu(g)
        return F.softplus(y)*self.mocu_scale

    def forward_erm(self, belief_graph: dict, xi: tuple[int,int]) -> torch.Tensor:
        node = torch.as_tensor(belief_graph["node_feats"], dtype=torch.float)
        edge = torch.as_tensor(belief_graph["edge_feats"], dtype=torch.float)
        idx  = torch.as_tensor(belief_graph["edge_index"], dtype=torch.long)
        g = self._aggregate(node, edge, idx)
        N = node.shape[0]; i,j = xi
        cand = torch.tensor([[i/(N-1+1e-9), j/(N-1+1e-9)]], dtype=torch.float)
        z = torch.cat([g, cand], dim=-1)
        return F.softplus(self.head_erm(z))

    def forward_sync(self, A_min: np.ndarray, a_ctrl: float, belief_graph: dict) -> torch.Tensor:
        node = torch.as_tensor(belief_graph["node_feats"], dtype=torch.float)
        edge = torch.as_tensor(belief_graph["edge_feats"], dtype=torch.float)
        idx  = torch.as_tensor(belief_graph["edge_index"], dtype=torch.long)
        g = self._aggregate(node, edge, idx)
        u = torch.tensor([[a_ctrl]], dtype=torch.float)
        z = torch.cat([g, u], dim=-1)
        return torch.sigmoid(self.head_sync(z))  # prob(sync)
