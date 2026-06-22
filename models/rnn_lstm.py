"""RNN + LSTM matching the paper's architecture (Table 4).

  LSTM units per layer  : 64
  Number of hidden layers : 2
  Optimizer             : Adam, lr = 0.001
  Epochs                : 20
  Batch size            : 128

Processes each sensor node's temporal sequence independently through the LSTM,
mirroring the paper's per-record sequential model without graph structure.

Input  : x  [B, T, N, F]
Output : logits [B, N]
"""
import torch
import torch.nn as nn


class RNNLSTMModel(nn.Module):
    def __init__(self, n_features, lstm_units=64, lstm_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, lstm_units, lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Linear(lstm_units, 1)

    def forward(self, x):
        B, T, N, F = x.shape
        # treat each node's T-step window as an independent sequence
        x_nodes = x.permute(0, 2, 1, 3).reshape(B * N, T, F)  # [B*N, T, F]
        out, _ = self.lstm(x_nodes)                             # [B*N, T, units]
        logits = self.head(out[:, -1]).reshape(B, N)            # [B, N]
        return logits

    @torch.no_grad()
    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))
