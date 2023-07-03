import torch.nn as nn
import torch
from dataclasses import dataclass
from typing import Literal

# embedding을 하고 있지 않은데 적용 고려해볼 것.
class NN(nn.Module):
    def __init__(self, input_size, num_heads=3, num_layers=2):
        super(NN, self).__init__()
        self.input_size = input_size
        self.num_heads = num_heads

        self.blocks = nn.ModuleList([Block(input_size, num_heads) for _ in range(num_layers)])
        self.fc = nn.Linear(input_size, 1)


    def forward(self, x):
        '''
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            x: (batch_size, seq_len)
            attention_weights: (batch_size, num_layers, num_heads, seq_len, seq_len)
        '''
        attention_weights = []
        for block in self.blocks:
            x, aw = block(x)
            attention_weights.append(aw)
        output = self.fc(x)
        output = output.squeeze(2)
        attention_weights = torch.stack(attention_weights, dim=1)

        return output, attention_weights

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
            self.attention.append(Block(info.hidden_dim, info.num_heads))
        self.conv1d = nn.Conv1d(info.seq_len, 1, 1)
        self.output_layer = nn.Linear(info.hidden_dim, 1)

        self.device = info.device
        self.to(self.device)
    
    def _format(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
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
        self.softmax = nn.Softmax()

        self.attention = nn.ModuleList()
        for _ in range(info.num_blocks):
            self.attention.append(Block(info.hidden_dim, info.num_heads))
        self.output_layer = nn.Linear(info.hidden_dim, 1)

        self.device = info.device
        self.to(self.device)

    def _format(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
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

class Block(nn.Module):
    def __init__(self, input_size, num_heads):
        super(Block, self).__init__()
        self.input_size = input_size
        self.num_heads = num_heads

        self.fc1 = nn.Linear(input_size, 3 * input_size)
        self.mha = nn.MultiheadAttention(input_size, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.BatchNorm1d(input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(input_size, input_size)
        self.norm2 = nn.BatchNorm1d(input_size)

    def forward(self, x):
        '''
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            x: (batch_size, seq_len)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        '''
        seq_len = x.size(1)

        qkv = self.fc1(x)
        q = qkv[:, :, :self.input_size]
        k = qkv[:, :, self.input_size:self.input_size * 2]
        v = qkv[:, :, self.input_size * 2:]
        z, attention_weights = self.mha(q, k, v)
        x = x + z
        x = torch.stack([self.norm1(x[:, i]) for i in range(seq_len)], dim=1)
        z = self.fc2(x)
        z = self.relu(z)
        z = self.fc3(z)
        x = x + z
        output = torch.stack([self.norm1(x[:, i]) for i in range(seq_len)], dim=1)

        return output, attention_weights