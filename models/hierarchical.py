import torch
import torch.nn as nn

from models.transformer import (
    PositionalEncoding,
    EncoderLayer,
)


class HierarchicalGaslightingDetector(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_heads=4,
                 num_utterance_layers=1, num_conversation_layers=1,
                 dim_feedforward=512, num_classes=2, dropout=0.1,
                 max_length=50, max_utterances=6,):
        super().__init__()

        self.pad_idx = 0
        d_model = embedding_dim
        d_k = d_model // num_heads
        d_v = d_model // num_heads

        #token components
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=self.pad_idx,
        )
        self.token_positional = PositionalEncoding(d_model, max_len=max_length)

        #encoder for utternace(tokens)
        self.utterance_layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    n_heads=num_heads,
                    d_hidden=dim_feedforward,
                    d_k=d_k,
                    d_v=d_v,
                    dropout=dropout,
                )
                for _ in range(num_utterance_layers)
            ]
        )

        #encoder for conversation (utterance embeddings)
        self.conv_positional = PositionalEncoding(d_model, max_len=max_utterances)
        self.conv_layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    n_heads=num_heads,
                    d_hidden=dim_feedforward,
                    d_k=d_k,
                    d_v=d_v,
                    dropout=dropout,
                )
                for _ in range(num_conversation_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

        # 2 logit classifier head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, input_ids):
        #inputs
        B, T, L = input_ids.size()
        flat_ids = input_ids.view(B * T, L)

        #mask for token
        token_mask = (flat_ids != self.pad_idx)
        attn_mask = token_mask.unsqueeze(1)

        #encoder for utterance
        x = self.embedding(flat_ids)
        x = self.token_positional(x)
        x = self.dropout(x)

        for layer in self.utterance_layers:
            x = layer(x, attn_mask)

        non_pad = token_mask.float().unsqueeze(-1)
        summed = (x * non_pad).sum(dim=1)
        counts = non_pad.sum(dim=1).clamp(min=1.0)
        utt_repr = summed / counts 

        utt_repr = utt_repr.view(B, T, -1) 

        utt_non_pad = (input_ids != self.pad_idx).any(dim=2)
        conv_mask = utt_non_pad.unsqueeze(1)       

        #conversation encoder
        h = self.conv_positional(utt_repr)
        h = self.dropout(h)

        for layer in self.conv_layers:
            h = layer(h, conv_mask)

        mask_float = utt_non_pad.float().unsqueeze(-1)
        conv_sum = (h * mask_float).sum(dim=1)
        conv_count = mask_float.sum(dim=1).clamp(min=1.0)
        pooled = conv_sum / conv_count

        logits = self.classifier(pooled)
        return logits
