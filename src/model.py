import torch
from torch import nn


class DigitsClassifier(nn.Module):
    def __init__(self, n_layers=1, hidden=64, dropout=0.0):
        super().__init__()

        layers = []
        input_size = 64

        # Создаём скрытые слои
        for _ in range(n_layers):
            layers.append(nn.Linear(input_size, hidden))
            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            input_size = hidden

        # Выходной слой: 10 классов (цифры 0–9)
        layers.append(nn.Linear(input_size, 10))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
