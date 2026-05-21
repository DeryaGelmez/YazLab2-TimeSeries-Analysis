import torch.nn as nn

from .cnn1d_model import CNN1DClassifier
from .gru_model import GRUClassifier
from .lstm_model import LSTMClassifier

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "lstm": LSTMClassifier,
    "gru": GRUClassifier,
    "cnn1d": CNN1DClassifier,
}


def build_model(name: str, input_size: int) -> nn.Module:
    if name not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Supported models: {supported}")

    model_cls = MODEL_REGISTRY[name]
    return model_cls(input_size=input_size)
