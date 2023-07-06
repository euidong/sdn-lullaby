from typing import Literal
from dataclasses import dataclass

import torch.nn as nn

from src.model.base import SelfAttentionBlock


@dataclass
class PPOValueInfo:
    in_dim: int
    hidden_dim: int
    seq_len: int
    num_blocks: int
    num_heads: int
    device: Literal['cpu', 'cuda']

class PPOValue(nn.Module):
    def __init__(self, info: PPOValueInfo):
        super(PPOValue, self).__init__()
        self.info = info

        self.input_layer = nn.Linear(info.in_dim, info.hidden_dim)
        self.relu = nn.ReLU()

        self.attention = nn.ModuleList()
        for _ in range(info.num_blocks):
            self.attention.append(SelfAttentionBlock(info.hidden_dim, info.num_heads))
        self.conv1d = nn.Conv1d(info.seq_len, 1, 1)
        self.output_layer = nn.Linear(info.hidden_dim, 1)

        self.device = info.device
        self.to(self.device)
    
    def _format(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        return x

    def forward(self, x):
        x = self._format(x)
        x = self.relu(self.input_layer(x))
        for block in self.attention:
            x, _ = block(x)
        x = self.conv1d(x)
        output = self.output_layer(x)
        output = output.squeeze()
        return output

@dataclass
class PPOPolicyInfo:
    in_dim: int
    hidden_dim: int
    out_dim: int
    num_blocks: int
    num_heads: int
    device: Literal['cpu', 'cuda']

class PPOPolicy(nn.Module):
    def __init__(self, info: PPOPolicyInfo):
        super(PPOPolicy, self).__init__()
        self.info = info

        self.input_layer = nn.Linear(info.in_dim, info.hidden_dim)
        self.relu = nn.ReLU()

        self.attention = nn.ModuleList()
        for _ in range(info.num_blocks):
            self.attention.append(SelfAttentionBlock(info.hidden_dim, info.num_heads))
        self.output_layer = nn.Linear(info.hidden_dim, 1)

        self.device = info.device
        self.to(self.device)

    def _format(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        return x

    def forward(self, x):
        x = self._format(x)
        x = self.relu(self.input_layer(x))
        for block in self.attention:
            x, _ = block(x)
        output = self.output_layer(x)
        output = output.squeeze()
        return output
