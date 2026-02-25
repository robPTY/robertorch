import math
from typing import Dict

import torch
import torch.nn as nn 
from torch import Tensor

class MultiHeadAttention(nn.Module):
    def __init__(self, config: Dict) -> None:
        '''Initialize a standard MultiHeadAttention block following 2017 Vaswani et. al.'''
        super().__init__()
        self.cfg = config
        self.num_heads = config["N_HEADS"]
        self.d_model = config["D_MODEL"]
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.dk = config["D_MODEL"] // config["N_HEADS"]  # 512/8 = 64
        self.Wq = nn.Linear(config["D_MODEL"], config["D_MODEL"], bias=config["QKV_BIAS"])
        self.Wk = nn.Linear(config["D_MODEL"], config["D_MODEL"], bias=config["QKV_BIAS"])
        self.Wv = nn.Linear(config["D_MODEL"], config["D_MODEL"], bias=config["QKV_BIAS"])
        self.Wo = nn.Linear(config["D_MODEL"], config["D_MODEL"], bias=config["QKV_BIAS"])

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
        '''Project inputs into multiple heads, and compute attention scores with optional masking.'''
        batch, q_len, d_emb = q.shape  # (B, T, D)
        k_len = k.shape[1]
        v_len = v.shape[1]

        # Initial projection
        queries = self.Wq(q)  # (B, T, 512) x (512, 512) -> (B, T, 512)
        keys = self.Wk(k)  # (B, T, 512) x (512, 512) -> (B, T, 512)
        values = self.Wv(v)  # (B, T, 512) x (512, 512) -> (B, T, 512)

        # Reshape for n heads (B, T, H, D//H) -> (B, H, T, D//H)
        queries = queries.view(batch, q_len, self.num_heads, self.dk).transpose(1, 2)
        keys = keys.view(batch, k_len, self.num_heads, self.dk).transpose(1, 2)
        values = values.view(batch, v_len, self.num_heads, self.dk).transpose(1, 2)

        # Dot prod of queries and keys
        attn_scores = queries @ keys.transpose(2, 3)  # (B, H, T, 64) x (B, H, 64, T)
        scaled_scores = attn_scores / math.sqrt(self.dk)

        # Apply masking before softmax
        if mask is not None:
            scaled_scores = scaled_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(scaled_scores, dim=-1) # (B, H, T, T)

        # Dot prod of "weights" and values
        attention = attn_weights @ values  # (B, H, T, 64) x (B, H, T, 64)

        # (B, H, T, D//H) -> (B, T, H, D//H) -> (B, T, D)
        attention = attention.transpose(1, 2).contiguous().view(batch, q_len, d_emb)

        attn_out = self.Wo(attention) # (B, T, D) x (D, D)
        return attn_out