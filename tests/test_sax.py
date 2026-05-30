import numpy as np
import pytest
from src.automata.sax import apply_sax, generate_breakpoints


class TestGenerateBreakpoints:

    def test_alphabet_size_2_returns_one_breakpoint(self):
        bp = generate_breakpoints(2)
        assert len(bp) == 1
        assert np.isclose(bp[0], 0.0, atol=1e-6)

    def test_alphabet_size_3_returns_two_breakpoints(self):
        bp = generate_breakpoints(3)
        assert len(bp) == 2

    def test_breakpoints_are_sorted(self):
        bp = generate_breakpoints(5)
        assert list(bp) == sorted(bp)

    def test_alphabet_size_1_raises(self):
        with pytest.raises(ValueError):
            generate_breakpoints(1)

    def test_breakpoints_count_is_alphabet_minus_one(self):
        for size in [2, 3, 4, 5, 6]:
            bp = generate_breakpoints(size)
            assert len(bp) == size - 1


class TestApplySAX:

    def test_output_length_equals_input_length(self):
        paa_values = np.array([0.0, 1.0, -1.0])
        result = apply_sax(paa_values, alphabet_size=3)
        assert len(result) == 3

    def test_output_is_string(self):
        paa_values = np.array([0.0])
        result = apply_sax(paa_values, alphabet_size=3)
        assert isinstance(result, str)

    def test_symbols_within_alphabet(self):
        paa_values = np.array([0.0, 1.0, -1.0, 0.5])
        result = apply_sax(paa_values, alphabet_size=3)
        valid_chars = set("abc")
        assert all(c in valid_chars for c in result)

    def test_large_positive_value_maps_to_last_symbol(self):
        paa_values = np.array([100.0])
        result = apply_sax(paa_values, alphabet_size=3)
        assert result == "c"

    def test_large_negative_value_maps_to_first_symbol(self):
        paa_values = np.array([-100.0])
        result = apply_sax(paa_values, alphabet_size=3)
        assert result == "a"

    def test_alphabet_size_determines_symbol_range(self):
        paa_values = np.array([100.0])
        result = apply_sax(paa_values, alphabet_size=5)
        assert result == "e"

    def test_monotonically_increasing_input(self):
        paa_values = np.array([-2.0, -0.5, 0.5, 2.0])
        result = apply_sax(paa_values, alphabet_size=3)
        symbols = list(result)
        assert symbols == sorted(symbols)

    def test_all_same_values_produce_same_symbol(self):
        paa_values = np.array([0.0, 0.0, 0.0])
        result = apply_sax(paa_values, alphabet_size=3)
        assert len(set(result)) == 1
