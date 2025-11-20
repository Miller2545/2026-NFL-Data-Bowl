# model_torch.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LSTMFrameConditionedModel(nn.Module):
    """
    PyTorch version of your Keras model:
      - Input 1: sequence [batch, time, features]
      - Input 2: scalar time index [batch, 1]
      - Output: (x, y) coordinates [batch, 2]
    """

    def __init__(self, n_in_steps: int, n_features: int, hidden_units: int = 64):
        super().__init__()
        self.n_in_steps = n_in_steps
        self.n_features = n_features
        self.hidden_units = hidden_units

        # LSTM stack
        self.lstm1 = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_units,
            batch_first=True,
            bidirectional=False,
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden_units,
            hidden_size=hidden_units,
            batch_first=True,
            bidirectional=False,
        )

        # Time branch (scalar input -> small embedding)
        self.time_fc = nn.Linear(1, 16)

        # MLP head after concatenation
        self.fc1 = nn.Linear(hidden_units + 16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 2)  # (x, y)

        self.dropout = nn.Dropout(0.2)

    def forward(self, seq_input: torch.Tensor, time_input: torch.Tensor) -> torch.Tensor:
        """
        seq_input: [batch, time, features]
        time_input: [batch, 1]
        returns: [batch, 2]
        """
        # LSTM encoder
        # We are not doing masking/packing here; zero-padded frames at the start
        # are treated as part of the sequence, which is usually acceptable.
        x, _ = self.lstm1(seq_input)
        x, _ = self.lstm2(x)
        # Take last timestep representation: x[:, -1, :] -> [batch, hidden_units]
        x = x[:, -1, :]
        x = self.dropout(x)

        # Time branch
        t = F.relu(self.time_fc(time_input))

        # Concatenate [batch, hidden_units + 16]
        merged = torch.cat([x, t], dim=-1)

        # MLP head
        z = F.relu(self.fc1(merged))
        z = F.relu(self.fc2(z))
        out = self.fc_out(z)
        return out
    
class GRUFrameConditionedModel(nn.Module):
    """
    Same idea as LSTM model, but using GRUs.
    This will learn slightly different temporal dynamics → good for ensembles.
    """
    def __init__(self, n_in_steps: int, n_features: int, hidden_units: int = 64):
        super().__init__()
        self.gru1 = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_units,
            batch_first=True,
        )
        self.gru2 = nn.GRU(
            input_size=hidden_units,
            hidden_size=hidden_units,
            batch_first=True,
        )

        self.time_fc = nn.Linear(1, 16)

        self.fc1 = nn.Linear(hidden_units + 16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 2)

        self.dropout = nn.Dropout(0.2)

    def forward(self, seq_input: torch.Tensor, time_input: torch.Tensor) -> torch.Tensor:
        x, _ = self.gru1(seq_input)
        x, _ = self.gru2(x)
        x = x[:, -1, :]
        x = self.dropout(x)

        t = F.relu(self.time_fc(time_input))

        merged = torch.cat([x, t], dim=-1)
        z = F.relu(self.fc1(merged))
        z = F.relu(self.fc2(z))
        out = self.fc_out(z)
        return out

class TimePositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for time steps.
    shape in:  [batch, time, d_model]
    shape out: [batch, time, d_model]
    """
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, d_model]
        t = x.size(1)
        return x + self.pe[:, :t]


class TransformerFrameConditionedModel(nn.Module):
    """
    Frame-conditioned model with:
      - temporal Conv1D over features across time
      - Transformer encoder (pre-norm) over sequence
      - CLS token for pooling
      - separate time-conditioning branch
      - MLP head to predict (x, y)

    Inputs:
      seq_input : [B, T, F]
      time_input: [B, 1]
    Output:
      [B, 2]  (predicted x, y)
    """

    def __init__(
        self,
        n_in_steps: int,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        conv_kernel_size: int = 3,
    ):
        super().__init__()
        self.n_in_steps = n_in_steps
        self.n_features = n_features

        # --- Temporal Conv1D over time (local motion patterns) ---
        # Conv1d expects [B, C(in), T]; we use channels = features.
        padding = conv_kernel_size // 2  # keep length == n_in_steps
        self.temporal_conv = nn.Conv1d(
            in_channels=n_features,
            out_channels=n_features,
            kernel_size=conv_kernel_size,
            padding=padding,
        )

        # Project features -> transformer model dimension
        self.input_proj = nn.Linear(n_features, d_model)

        # CLS token + positional encoding (T + 1 for CLS)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_encoding = TimePositionalEncoding(d_model, max_len=n_in_steps + 1)

        # Pre-norm transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # we use [B, T, D]
            norm_first=True,    # pre-layernorm, usually more stable
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Time-conditioning branch
        self.time_fc = nn.Linear(1, 16)

        # MLP head after concatenation of CLS + time_encoding
        self.fc1 = nn.Linear(d_model + 16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 2)

        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_input: torch.Tensor, time_input: torch.Tensor) -> torch.Tensor:
        """
        seq_input : [B, T, F]
        time_input: [B, 1]
        """
        B, T, feat_dim = seq_input.shape 

        # ---- Temporal Conv1D over time ----
        x = seq_input.transpose(1, 2)      # [B, feat_dim, T]
        x = self.temporal_conv(x)          # [B, feat_dim, T]
        x = F.relu(x)                      
        x = x.transpose(1, 2)              # [B, T, feat_dim]

        # ---- Project into transformer d_model ----
        x = self.input_proj(x)             # [B, T, D]

        # ---- Add CLS token ----
        cls = self.cls_token.expand(B, 1, -1)   # [B, 1, D]
        x = torch.cat([cls, x], dim=1)          # [B, 1+T, D]

        # ---- Positional encoding & transformer ----
        x = self.pos_encoding(x)                # [B, 1+T, D]
        x = self.transformer_encoder(x)         # [B, 1+T, D]

        # Take CLS output as sequence summary
        cls_out = x[:, 0, :]                    # [B, D]
        cls_out = self.dropout(cls_out)

        # ---- Time-conditioning branch ----
        t_enc = F.relu(self.time_fc(time_input))   # [B, 16]

        # ---- Merge and predict (x, y) ----
        merged = torch.cat([cls_out, t_enc], dim=-1)  # [B, D+16]
        z = F.relu(self.fc1(merged))
        z = F.relu(self.fc2(z))
        out = self.fc_out(z)                         # [B, 2]

        return out
