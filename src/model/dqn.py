from typing import Literal
from dataclasses import dataclass

import torch.nn as nn

from src.model.base import SelfAttentionBlock


@dataclass
class DQNValueInfo:
    in_dim: int
    hidden_dim: int
    num_heads: int
    num_blocks: int
    device: Literal['cpu', 'cuda']

class DQNValue(nn.Module):
    def __init__(self, info: DQNValueInfo):
        super(DQNValue, self).__init__()
        self.info = info

        self.input_layer = nn.Linear(info.in_dim, info.hidden_dim)
        self.relu = nn.ReLU()

        self.attention = nn.ModuleList()
        for _ in range(info.num_blocks):
            self.attention.append(SelfAttentionBlock(info.hidden_dim, info.num_heads))
        self.output_layer = nn.Linear(info.hidden_dim, 1)

        self.device = info.device
        self.to(self.device)


    def forward(self, x):
        '''
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            x: (batch_size, seq_len)
        '''
        x = self._format(x)
        x = self.relu(self.input_layer(x))
        for block in self.attention:
            x, _ = block(x)
        output = self.output_layer(x)
        output = output.squeeze(2)
        return output

    def _format(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = x.to(self.device)
        return x
