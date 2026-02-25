from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float) -> None:
        '''Initialize a normalization layer with model dimension and eps value'''
        super().__init__()
        self.eps = eps  # For numerical stability and to avoid division by zero
        # nn.Parameter makes them learnable
        self.alpha = nn.Parameter(torch.ones(d_model))  # Multiplies the xj
        self.bias = nn.Parameter(torch.zeros(d_model))  # Gets added to xj

    def forward(self, x: Tensor) -> Tensor:
        '''Normalize input tensor using mean and variance, then scale + shift with learnable parameters.'''
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.std(dim=-1, unbiased=False, keepdim=True)
        return self.alpha * (x - mean) / torch.sqrt(variance + self.eps) + self.bias

class ResidualLayer(nn.Module):
    def __init__(self, config: Dict):
        '''Initialize a residual layer with a config dict for dropout and a layer norm.'''
        super().__init__()
        self.cfg = config
        self.dropout = nn.Dropout(config["DROPOUT"])
        self.norm = LayerNorm(config["D_MODEL"], config["EPS"])

    def forward(self, x: Tensor, sublayer: nn.Module) -> Tensor:
        '''Execute the Pre-Norm residual connection logic: Norm(x + Dropout(Sublayer(x))).'''
        return self.norm(x + self.dropout(sublayer(x)))
    
class LinearLayer(nn.Module):
    def __init__(self, config: Dict) -> None:
        '''Initialize a linear projection layer with a config dict for model dims and size of embeddings'''
        super().__init__()
        self.cfg = config
        # So, in the manuscript they mention in section 3.4 that the pre-softmax transformation weights
        # are the same ones used in the embedding layers
        self.projection = nn.Linear(config["D_MODEL"], config["EMBEDDING_SIZE"])

    def forward(self, x: Tensor) -> Tensor:
        '''Perform a linear transformation to project the input tensor to the output embedding space.'''
        return self.projection(x) # (B, T, D) x (D, V) -> (B, T, V)