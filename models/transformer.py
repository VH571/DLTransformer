import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Masking Utilities
def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Sublayers Multi-Head Attention and Feed Forward
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, dropout):
        super().__init__()

        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_heads * d_v, bias=False)

        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = d_k ** 0.5

    def forward(self, x, mask=None):
        B, L, _ = x.size()

        q = self.w_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.n_heads, self.d_v).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v) 
        context = context.transpose(1, 2).contiguous().view(B, L, -1)

        out = self.fc(context)
        out = self.layer_norm(x + out)
        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.fc2(F.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.layer_norm(residual + x)
        return x


# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_hidden, d_k, d_v, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_hidden, dropout)

    def forward(self, x, mask):
        x = self.self_attn(x, mask)
        x = self.ffn(x)
        return x


# Transformer Gaslighting Classifier
class TransformerGaslightingDetector(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
        dim_feedforward=512,
        num_classes=2,
        dropout=0.1,
        max_length=50
    ):
        super().__init__()

        self.pad_idx = 0
        d_model = embedding_dim
        d_k = d_model // num_heads
        d_v = d_model // num_heads

        # Embedding positional encoding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=self.pad_idx
        )
        self.positional = PositionalEncoding(d_model, max_len=max_length)
        self.dropout = nn.Dropout(dropout)

        # Encoder stack
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                n_heads=num_heads,
                d_hidden=dim_feedforward,
                d_k=d_k,
                d_v=d_v,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, input_ids):
        mask = get_pad_mask(input_ids, self.pad_idx) 

        x = self.embedding(input_ids)       
        x = self.positional(x) 
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)
        non_pad = (input_ids != self.pad_idx).float().unsqueeze(-1)
        pooled = (x * non_pad).sum(dim=1) / non_pad.sum(dim=1)

        logits = self.classifier(pooled)
        return logits
