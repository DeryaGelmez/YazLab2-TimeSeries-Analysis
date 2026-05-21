import torch
import torch.nn as nn

from config.config import LSTM_DROPOUT, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=LSTM_DROPOUT,
        )
        self.fc = nn.Linear(LSTM_HIDDEN_SIZE, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.fc(last_hidden)
