import torch.nn as nn
import torch

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