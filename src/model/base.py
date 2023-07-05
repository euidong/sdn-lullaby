import torch
import torch.nn as nn


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
        batch_len = x.size(0)
        seq_len = x.size(1)

        qkv = self.fc1(x)
        q = qkv[:, :, :self.input_size]
        k = qkv[:, :, self.input_size:self.input_size * 2]
        v = qkv[:, :, self.input_size * 2:]
        z, attention_weights = self.mha(q, k, v)
        x = x + z
        
        x = torch.stack([self.norm1(x[:, i]) if batch_len > 1 else x[:, i] for i in range(seq_len)], dim=1)
        z = self.fc2(x)
        z = self.relu(z)
        z = self.fc3(z)
        x = x + z
        output = torch.stack([self.norm2(x[:, i]) if batch_len > 1 else x[:, i] for i in range(seq_len)], dim=1)

        return output, attention_weights
