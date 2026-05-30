import numpy as np
import pytest
from src.automata.paa import apply_paa


class TestApplyPAA:

    def test_output_length_equals_n_segments(self):
        ts = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = apply_paa(ts, n_segments=3)
        assert len(result) == 3

    def test_single_segment_returns_mean(self):
        ts = np.array([1.0, 2.0, 3.0, 4.0])
        result = apply_paa(ts, n_segments=1)
        assert len(result) == 1
        assert np.isclose(result[0], 2.5)

    def test_segment_values_are_averages(self):
        ts = np.array([1.0, 3.0, 5.0, 7.0])
        result = apply_paa(ts, n_segments=2)
        assert np.isclose(result[0], 2.0)  # mean(1, 3)
        assert np.isclose(result[1], 6.0)  # mean(5, 7)

    def test_n_segments_equals_length_returns_original(self):
        ts = np.array([1.0, 2.0, 3.0])
        result = apply_paa(ts, n_segments=3)
        np.testing.assert_array_almost_equal(result, ts)

    def test_non_divisible_length_handled(self):
        ts = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_paa(ts, n_segments=3)
        assert len(result) == 3

    def test_zero_segments_raises(self):
        ts = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            apply_paa(ts, n_segments=0)

    def test_negative_segments_raises(self):
        ts = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            apply_paa(ts, n_segments=-1)

    def test_more_segments_than_length_raises(self):
        ts = np.array([1.0, 2.0])
        with pytest.raises(ValueError):
            apply_paa(ts, n_segments=5)

    def test_2d_input_raises(self):
        ts = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError):
            apply_paa(ts, n_segments=2)

    def test_list_input_accepted(self):
        ts = [1.0, 2.0, 3.0, 4.0]
        result = apply_paa(ts, n_segments=2)
        assert len(result) == 2

    def test_all_same_values(self):
        ts = np.array([5.0, 5.0, 5.0, 5.0])
        result = apply_paa(ts, n_segments=2)
        np.testing.assert_array_almost_equal(result, [5.0, 5.0])
