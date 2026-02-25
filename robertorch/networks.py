from typing import Dict 

import torch.nn as nn
from torch import Tensor

class MLPBlockGeLU(nn.Module):
    def __init__(self, cfg: Dict[str, int]) -> None:
        '''Initialize a 2-layered MLP block with GeLU activation function and dropout'''
        super().__init__() 
        self.W1 = nn.Linear(cfg['D'], cfg['MLP_SIZE'])
        self.W2 = nn.Linear(cfg['MLP_SIZE'], cfg['D'])
        self.m = nn.GELU()
        self.dropout = nn.Dropout(cfg['p'])
    
    def forward(self, x: Tensor) -> Tensor:
        '''Return a tensor being passed through two MLP layers and GELU non-linearity'''
        xForward = self.W1(x)
        xGelud = self.m(xForward)
        return self.W2(self.dropout(xGelud))