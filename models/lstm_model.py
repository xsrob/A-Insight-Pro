"""
A-Insight Pro
LSTM Stock Prediction Model V2.0
- Bidirectional LSTM with Multi-Head Attention
- LayerNorm (replaces BatchNorm) for time-series stability
- Residual connections with proper projection
- Huber Loss for robustness
- Optional regime prediction head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention over LSTM output sequence."""

    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # q,k,v: [batch, heads, seq_len, head_dim]

        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # [batch, heads, seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(attn_output)


class StockLSTM(nn.Module):
    """
    Bidirectional LSTM + Attention for stock return prediction.

    Architecture:
        Input [batch, seq_len, n_features]
          -> LayerNorm (feature-wise, applied per time step)
          -> BiLSTM(hidden=128) -> Dropout
          -> BiLSTM(hidden=64) -> Dropout
          -> MultiHeadAttention(4 heads)
          -> Residual + LayerNorm
          -> GlobalAvgPool + GlobalMaxPool (concatenated)
          -> Dense(128) -> ReLU -> Dropout
          -> Dense(64) -> ReLU -> Dropout
          -> Dense(1)  [predicted 5-day return]

    V2.0 Changes:
      - LayerNorm replaces BatchNorm (correct for time series)
      - Removed dead code in residual projection
      - GlobalAvgPool + GlobalMaxPool for richer aggregation
      - Proper Xavier init with gain for ReLU
    """

    def __init__(self, n_features, hidden_size_1=128, hidden_size_2=64,
                 num_layers=2, dropout=0.3, attention_heads=4):
        super().__init__()
        self.n_features = n_features

        # ---- Input normalization (LayerNorm over feature dim) ----
        # For time series, LayerNorm is preferred over BatchNorm because:
        # - Each sample is independently normalized (no batch dependency)
        # - Works correctly at inference time with batch_size=1
        self.input_norm = nn.LayerNorm(n_features)

        # ---- First BiLSTM layer ----
        lstm1_output = hidden_size_1 * 2  # bidirectional
        self.lstm1 = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size_1,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(lstm1_output)  # LayerNorm after LSTM1

        # ---- Second BiLSTM layer ----
        self.lstm2 = nn.LSTM(
            input_size=lstm1_output,
            hidden_size=hidden_size_2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout2 = nn.Dropout(dropout)

        lstm2_output = hidden_size_2 * 2  # bidirectional

        # ---- Multi-Head Attention ----
        self.attention = MultiHeadAttention(lstm2_output, num_heads=attention_heads, dropout=dropout)

        # ---- Residual connection ----
        # Always project — residual_proj maps LSTM2 output to attention space.
        # (Previously had `if True else nn.Identity()` which was dead code)
        self.residual_proj = nn.Linear(lstm2_output, lstm2_output)
        self.layer_norm = nn.LayerNorm(lstm2_output)

        # ---- Pooling: GlobalAvgPool + GlobalMaxPool for richer aggregation ----
        # Concatenate mean and max → 2 * lstm2_output
        pooled_dim = lstm2_output * 2

        # ---- Output heads ----
        self.fc1 = nn.Linear(pooled_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(0.2)

        # ---- Initialize weights ----
        self._init_weights()

    def _init_weights(self):
        """Xavier init for Linear layers, orthogonal init for LSTM weights."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                if 'lstm' in name:
                    # Orthogonal init is better for RNN/LSTM gates
                    nn.init.orthogonal_(param)
                else:
                    # Xavier uniform for linear layers
                    nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 for LSTM (helps with long-term memory)
                if 'lstm' in name and param.dim() > 0:
                    n = param.size(0)
                    # LSTM bias: [input_gate, forget_gate, cell_gate, output_gate] * 2 (bidirectional)
                    # Set forget gate biases to 1
                    param.data[n // 4:n // 2].fill_(1.0)

    def forward(self, x):
        # x: [batch, seq_len, n_features]
        batch_size, seq_len, n_features = x.shape

        # LayerNorm over feature dimension (applied per time step)
        # Reshape to [batch * seq_len, n_features] for LayerNorm, then back
        x_reshaped = x.reshape(-1, n_features)
        x_norm = self.input_norm(x_reshaped).reshape(batch_size, seq_len, n_features)

        # BiLSTM 1
        lstm_out1, _ = self.lstm1(x_norm)  # [batch, seq_len, 2*hidden_size_1]
        lstm_out1 = self.norm1(lstm_out1)   # LayerNorm
        lstm_out1 = self.dropout1(lstm_out1)

        # BiLSTM 2
        lstm_out2, _ = self.lstm2(lstm_out1)  # [batch, seq_len, 2*hidden_size_2]
        # lstm_out2 = self.dropout2(lstm_out2)  # Apply dropout AFTER residual

        # Attention
        attn_out = self.attention(lstm_out2)  # [batch, seq_len, lstm2_output]

        # Residual connection
        residual = self.residual_proj(lstm_out2)
        combined = self.layer_norm(attn_out + residual)
        combined = self.dropout2(combined)

        # Global Average Pooling + Global Max Pooling
        avg_pooled = combined.mean(dim=1)  # [batch, lstm2_output]
        max_pooled = combined.max(dim=1).values  # [batch, lstm2_output]
        pooled = torch.cat([avg_pooled, max_pooled], dim=-1)  # [batch, 2*lstm2_output]

        # Dense layers
        out = self.relu(self.fc1(pooled))
        out = self.dropout_fc(out)
        out = self.relu(self.fc2(out))
        out = self.dropout_fc(out)
        out = self.fc_out(out)  # [batch, 1]

        return out.squeeze(-1)


class HuberLoss(nn.Module):
    """Huber loss - MSE for small errors, MAE for large errors."""

    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target):
        diff = pred - target
        abs_diff = torch.abs(diff)
        quadratic = 0.5 * diff ** 2
        linear = self.delta * (abs_diff - 0.5 * self.delta)
        loss = torch.where(abs_diff <= self.delta, quadratic, linear)
        return loss.mean()


def create_model(n_features, config=None):
    """Factory function to create the StockLSTM model."""
    if config is None:
        config = {}
    return StockLSTM(
        n_features=n_features,
        hidden_size_1=config.get("hidden_size_1", 128),
        hidden_size_2=config.get("hidden_size_2", 64),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.3),
        attention_heads=config.get("attention_heads", 4)
    )
