"""GAT-LSTM: the main model.

Graph Attention captures the spatial relationship between sensor nodes at each
timestep; the LSTM then models the temporal evolution of each node's
spatially-aware embedding. Output is a per-node fire probability.

Input  : x   [B, T, N, F]   batch, time, nodes, features
         adj [N, N]         binary adjacency (k-NN graph over node coords)
Output : logits [B, N]      one fire logit per node (apply sigmoid -> prob)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.2, concat=True):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.concat = concat
        self.W = nn.Parameter(torch.empty(n_heads, in_dim, out_dim))
        self.a_src = nn.Parameter(torch.empty(n_heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(n_heads, out_dim))
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(-1))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(-1))
        self.leaky = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, adj):
        # h: [B, N, in_dim]
        B, N, _ = h.shape
        Wh = torch.einsum("bni,hio->bhno", h, self.W)        # [B,H,N,out]
        e_src = torch.einsum("bhno,ho->bhn", Wh, self.a_src)  # [B,H,N]
        e_dst = torch.einsum("bhno,ho->bhn", Wh, self.a_dst)  # [B,H,N]
        e = self.leaky(e_src.unsqueeze(-1) + e_dst.unsqueeze(-2))  # [B,H,N,N]
        mask = (adj > 0).unsqueeze(0).unsqueeze(0)
        e = e.masked_fill(~mask, float("-inf"))
        alpha = self.drop(F.softmax(e, dim=-1))
        out = torch.einsum("bhnm,bhmo->bhno", alpha, Wh)      # [B,H,N,out]
        if self.concat:
            out = out.permute(0, 2, 1, 3).reshape(B, N, self.n_heads * self.out_dim)
            return F.elu(out)
        return out.mean(dim=1)                                # [B,N,out]


class GATLSTM(nn.Module):
    def __init__(self, n_features, gat_hidden=64, gat_heads=4,
                 lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super().__init__()
        self.gat1      = GraphAttentionLayer(n_features, gat_hidden, gat_heads, dropout, concat=True)
        self.gat2      = GraphAttentionLayer(gat_hidden * gat_heads, gat_hidden, 1, dropout, concat=False)
        self.norm_gat  = nn.LayerNorm(gat_hidden)
        self.lstm      = nn.LSTM(gat_hidden, lstm_hidden, lstm_layers,
                                 batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.norm_lstm = nn.LayerNorm(lstm_hidden)
        self.head      = nn.Sequential(
            nn.Linear(lstm_hidden, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x, adj):
        B, T, N, _ = x.shape
        frames = []
        for t in range(T):
            h = self.gat1(x[:, t], adj)
            h = self.gat2(h, adj)           # [B,N,gat_hidden]
            h = self.norm_gat(h)            # stabilise across nodes
            frames.append(h)
        s = torch.stack(frames, dim=1)      # [B,T,N,gat_hidden]
        s = s.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        out, _ = self.lstm(s)               # [B*N,T,lstm_hidden]
        last   = self.norm_lstm(out[:, -1]) # LayerNorm on final step
        logits = self.head(last).reshape(B, N)
        return logits

    @torch.no_grad()
    def predict_proba(self, x, adj):
        return torch.sigmoid(self.forward(x, adj))
