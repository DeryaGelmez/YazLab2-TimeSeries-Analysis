import numpy as np
from scipy.stats import norm


def generate_breakpoints(alphabet_size):
   

    if alphabet_size < 2:
        raise ValueError("alphabet_size must be at least 2.")

    breakpoints = norm.ppf(
        np.linspace(
            0,
            1,
            alphabet_size + 1
        )[1:-1]
    )

    return breakpoints


def apply_sax(paa_values, alphabet_size):
   

    paa_values = np.asarray(paa_values)

    breakpoints = generate_breakpoints(alphabet_size)

    alphabet = [chr(i) for i in range(97, 97 + alphabet_size)]

    sax_symbols = []

    for value in paa_values:

        symbol_index = np.searchsorted(
            breakpoints,
            value
        )

        sax_symbols.append(
            alphabet[symbol_index]
        )

    return "".join(sax_symbols)