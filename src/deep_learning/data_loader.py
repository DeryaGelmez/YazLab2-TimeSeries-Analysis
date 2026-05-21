import random
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

from config.config import (
    BATADAL_TEST_SCALED,
    BATADAL_TRAIN_SCALED,
    BATADAL_VAL_SCALED,
    SKAB_TEST_SCALED,
    SKAB_TRAIN_SCALED,
)

LabelMode = Literal["last", "any"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TimeSeriesWindowDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int,
        stride: int,
        label_mode: LabelMode = "last",
    ) -> None:
        if X.ndim != 2:
            raise ValueError("X must be a 2D array with shape [N, F].")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")

        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.seq_len = seq_len
        self.stride = stride
        self.label_mode = label_mode
        self.starts = list(range(0, len(self.X) - seq_len + 1, stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[index]
        end = start + self.seq_len
        window_x = self.X[start:end]
        window_y = self.y[start:end]

        if self.label_mode == "any":
            label = float(window_y.max() > 0)
        else:
            label = float(window_y[-1])

        return torch.from_numpy(window_x), torch.tensor(label, dtype=torch.float32)


def load_skab_multivariate() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_df = pd.read_csv(SKAB_TRAIN_SCALED)
    test_df = pd.read_csv(SKAB_TEST_SCALED)
    df = pd.concat([train_df, test_df], ignore_index=True)

    meta_cols = {"anomaly", "source_file"}
    feature_cols = [col for col in df.columns if col not in meta_cols]

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["anomaly"].to_numpy(dtype=np.float32)
    groups = df["source_file"].to_numpy()

    return X, y, groups


def make_skab_groupkfold_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    gkf = GroupKFold(n_splits=n_splits)
    splits: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for train_val_idx, test_idx in gkf.split(X, y, groups):
        sorted_train_val = np.sort(train_val_idx)
        val_size = int(len(sorted_train_val) * 0.2)

        if val_size == 0 and len(sorted_train_val) > 1:
            val_size = 1

        if val_size > 0:
            train_idx = sorted_train_val[:-val_size]
            val_idx = sorted_train_val[-val_size:]
        else:
            train_idx = sorted_train_val
            val_idx = np.array([], dtype=int)

        splits.append((train_idx, val_idx, test_idx))

    return splits


def load_batadal_multivariate() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    train_df = pd.read_csv(BATADAL_TRAIN_SCALED)
    val_df = pd.read_csv(BATADAL_VAL_SCALED)
    test_df = pd.read_csv(BATADAL_TEST_SCALED)

    meta_cols = {"anomaly", "DATETIME"}

    def _split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        feature_cols = [col for col in df.columns if col not in meta_cols]
        X = df[feature_cols].to_numpy(dtype=np.float32)
        y = df["anomaly"].to_numpy(dtype=np.float32)
        return X, y

    X_tr, y_tr = _split(train_df)
    X_val, y_val = _split(val_df)
    X_te, y_te = _split(test_df)

    return X_tr, y_tr, X_val, y_val, X_te, y_te


def build_dataloaders(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    seq_len: int,
    stride: int,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = TimeSeriesWindowDataset(X_tr, y_tr, seq_len, stride)
    val_dataset = TimeSeriesWindowDataset(X_val, y_val, seq_len, stride)
    test_dataset = TimeSeriesWindowDataset(X_te, y_te, seq_len, stride)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader
