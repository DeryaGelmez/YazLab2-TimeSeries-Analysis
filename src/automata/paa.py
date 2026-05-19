import numpy as np


def apply_paa(time_series, n_segments):
    

    time_series = np.asarray(time_series, dtype=float)

    if time_series.ndim != 1:
        raise ValueError("PAA only supports one-dimensional time series.")

    if n_segments <= 0:
        raise ValueError("n_segments must be greater than 0.")

    if n_segments > len(time_series):
        raise ValueError("n_segments cannot be greater than time series length.")

    segments = np.array_split(time_series, n_segments)

    paa_values = np.array([
        segment.mean()
        for segment in segments
    ])

    return paa_values