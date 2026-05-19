def extract_patterns(symbolic_sequence, window_size):

    if not isinstance(symbolic_sequence, str):
        raise TypeError("symbolic_sequence must be a string.")

    if window_size <= 0:
        raise ValueError("window_size must be greater than 0.")

    if window_size > len(symbolic_sequence):
        raise ValueError("window_size cannot be greater than sequence length.")

    patterns = [
        symbolic_sequence[i:i + window_size]
        for i in range(len(symbolic_sequence) - window_size + 1)
    ]

    return patterns


def get_unique_states(patterns):

    return set(patterns)