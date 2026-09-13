import torch
import torch.nn as nn


class NN5Layer(nn.Module):
    def __init__(self, feature_num, label_num, hidden_size):
        super().__init__()

        self.feature_num = feature_num
        self.label_num = label_num
        self.hidden_size = hidden_size

        self.linears = nn.ModuleList([
            nn.Linear(feature_num, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, label_num)
        ])

    def forward(self, x):
        for i, linear in enumerate(self.linears):
            x = linear(x)
            if i < len(self.linears) - 1:
                x = torch.relu(x)
        return x

    def initialize(self):
        """Initialize linear-layer weights and biases."""

        for i, linear in enumerate(self.linears):
            if i < len(self.linears) - 1:
                nn.init.kaiming_normal_(
                    linear.weight,
                    nonlinearity="relu"
                )
            else:
                nn.init.kaiming_normal_(linear.weight, nonlinearity='linear')

            if linear.bias is not None:
                nn.init.zeros_(linear.bias)