import torch
import torch.nn as nn
import ipdb

class nn_5layer(nn.Module):
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
        for i, m in enumerate(self.linears):
            if isinstance(m, nn.Linear):
                if i < len(self.linears) - 1:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)