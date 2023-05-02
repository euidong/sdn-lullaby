import torch.nn as nn
import torch


class NN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(NN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2 * num_layers, 2 * num_layers)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2 * num_layers, num_layers)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
