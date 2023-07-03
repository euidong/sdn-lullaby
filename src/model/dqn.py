import torch
import torch.nn as nn

from src.model.base import Block


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
