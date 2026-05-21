import torch
import torch.nn as nn

from config.config import CNN1D_DROPOUT, CNN1D_KERNEL_SIZE, CNN1D_NUM_FILTERS


class CNN1DClassifier(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        padding = CNN1D_KERNEL_SIZE // 2

        self.block1 = nn.Sequential(
            nn.Conv1d(input_size, CNN1D_NUM_FILTERS, kernel_size=CNN1D_KERNEL_SIZE, padding=padding),
            nn.ReLU(),
            nn.BatchNorm1d(CNN1D_NUM_FILTERS),
            nn.MaxPool1d(kernel_size=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(CNN1D_NUM_FILTERS, CNN1D_NUM_FILTERS, kernel_size=CNN1D_KERNEL_SIZE, padding=padding),
            nn.ReLU(),
            nn.BatchNorm1d(CNN1D_NUM_FILTERS),
            nn.MaxPool1d(kernel_size=2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(CNN1D_DROPOUT)
        self.fc = nn.Linear(CNN1D_NUM_FILTERS, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)
