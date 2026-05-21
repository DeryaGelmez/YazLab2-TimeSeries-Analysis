from types import ModuleType
from typing import Callable

import numpy as np

ScenarioFn = Callable[..., np.ndarray]


def apply_original(X_test: np.ndarray) -> np.ndarray:
    return X_test


def apply_gaussian_noise(X_test: np.ndarray, std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=std, size=X_test.shape)
    return X_test + noise.astype(X_test.dtype, copy=False)


def apply_unseen_drift(
    X_test: np.ndarray,
    drift_mean: float,
    drift_scale: float,
    seed: int,
) -> np.ndarray:
    del seed
    return X_test * drift_scale + drift_mean


SCENARIO_REGISTRY: dict[str, ScenarioFn] = {
    "original": apply_original,
    "noise": apply_gaussian_noise,
    "unseen": apply_unseen_drift,
}


def apply_scenario(
    name: str,
    X_test: np.ndarray,
    config: ModuleType,
    seed: int,
) -> np.ndarray:
    if name not in SCENARIO_REGISTRY:
        supported = ", ".join(sorted(SCENARIO_REGISTRY))
        raise ValueError(f"Unknown scenario '{name}'. Supported scenarios: {supported}")

    if name == "original":
        return apply_original(X_test)

    if name == "noise":
        return apply_gaussian_noise(X_test, config.DL_GAUSSIAN_NOISE_STD, seed)

    return apply_unseen_drift(
        X_test,
        config.DL_UNSEEN_DRIFT_MEAN,
        config.DL_UNSEEN_DRIFT_SCALE,
        seed,
    )
