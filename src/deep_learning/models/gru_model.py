import torch
import torch.nn as nn

from config.config import GRU_DROPOUT, GRU_HIDDEN_SIZE, GRU_NUM_LAYERS


class GRUClassifier(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=GRU_HIDDEN_SIZE,
            num_layers=GRU_NUM_LAYERS,
            batch_first=True,
            dropout=GRU_DROPOUT,
        )
        self.fc = nn.Linear(GRU_HIDDEN_SIZE, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        last_hidden = output[:, -1, :]
        return self.fc(last_hidden)
