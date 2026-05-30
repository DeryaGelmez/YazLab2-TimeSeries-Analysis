import pytest
from src.automata.pattern import extract_patterns, get_unique_states


class TestExtractPatterns:

    def test_basic_sliding_window(self):
        result = extract_patterns("abcde", window_size=3)
        assert result == ["abc", "bcd", "cde"]

    def test_window_equals_length_returns_one_pattern(self):
        result = extract_patterns("abc", window_size=3)
        assert result == ["abc"]

    def test_window_size_1_returns_individual_chars(self):
        result = extract_patterns("abc", window_size=1)
        assert result == ["a", "b", "c"]

    def test_output_count_is_correct(self):
        seq = "abcdef"
        window = 4
        result = extract_patterns(seq, window_size=window)
        assert len(result) == len(seq) - window + 1

    def test_non_string_input_raises(self):
        with pytest.raises(TypeError):
            extract_patterns(["a", "b", "c"], window_size=2)

    def test_zero_window_raises(self):
        with pytest.raises(ValueError):
            extract_patterns("abc", window_size=0)

    def test_negative_window_raises(self):
        with pytest.raises(ValueError):
            extract_patterns("abc", window_size=-1)

    def test_window_larger_than_sequence_raises(self):
        with pytest.raises(ValueError):
            extract_patterns("ab", window_size=5)

    def test_repeated_chars(self):
        result = extract_patterns("aaaa", window_size=2)
        assert result == ["aa", "aa", "aa"]

    def test_sax_sequence_example(self):
        sax = "aabbc"
        result = extract_patterns(sax, window_size=3)
        assert result == ["aab", "abb", "bbc"]


class TestGetUniqueStates:

    def test_returns_set(self):
        patterns = ["abc", "bcd", "abc"]
        result = get_unique_states(patterns)
        assert isinstance(result, set)

    def test_duplicates_removed(self):
        patterns = ["abc", "abc", "def"]
        result = get_unique_states(patterns)
        assert result == {"abc", "def"}

    def test_all_unique(self):
        patterns = ["ab", "bc", "cd"]
        result = get_unique_states(patterns)
        assert result == {"ab", "bc", "cd"}

    def test_single_pattern(self):
        result = get_unique_states(["xyz"])
        assert result == {"xyz"}

    def test_empty_list_returns_empty_set(self):
        result = get_unique_states([])
        assert result == set()
