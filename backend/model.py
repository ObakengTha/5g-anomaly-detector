import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                              * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerAnomalyDetector(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, d_ff: int = 256, n_classes: int = 2,
                 max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, input_ids, return_attention=False):
        pad_mask = (input_ids == 0)
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)

        attentions = []
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=pad_mask)
            if return_attention:
                _, attn_w = layer.self_attn(
                    x, x, x, key_padding_mask=pad_mask,
                    need_weights=True, average_attn_weights=True,
                )
                attentions.append(attn_w.detach())

        x = self.norm(x)
        cls_repr = x[:, 0, :]
        logits = self.classifier(cls_repr)

        if return_attention:
            return logits, attentions
        return logits


def load_model(state_dict_path: str, vocab_size: int, max_len: int,
               n_classes: int = 2, device: str = "cpu"):
    model = TransformerAnomalyDetector(vocab_size=vocab_size, max_len=max_len,
                                        n_classes=n_classes)
    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
