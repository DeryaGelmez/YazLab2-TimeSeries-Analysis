import pytest
from src.automata.levenshtein import levenshtein_distance, find_nearest_pattern


class TestLevenshteinDistance:

    def test_identical_strings_return_zero(self):
        assert levenshtein_distance("abc", "abc") == 0

    def test_empty_strings_return_zero(self):
        assert levenshtein_distance("", "") == 0

    def test_empty_vs_nonempty_returns_length(self):
        assert levenshtein_distance("", "abc") == 3
        assert levenshtein_distance("abc", "") == 3

    def test_single_insertion(self):
        assert levenshtein_distance("ab", "abc") == 1

    def test_single_deletion(self):
        assert levenshtein_distance("abc", "ab") == 1

    def test_single_substitution(self):
        assert levenshtein_distance("abc", "axc") == 1

    def test_known_example_aab_adc(self):
        # "aab" -> "adc": substitutions at positions 1 and 2 → distance = 2
        assert levenshtein_distance("aab", "adc") == 2

    def test_known_example_abc_bcc(self):
        # "abc" -> "bcc": 2 substitutions
        assert levenshtein_distance("abc", "bcc") == 2

    def test_symmetry(self):
        assert levenshtein_distance("cat", "cut") == levenshtein_distance("cut", "cat")

    def test_completely_different(self):
        assert levenshtein_distance("aaa", "bbb") == 3

    def test_single_char_strings(self):
        assert levenshtein_distance("a", "b") == 1
        assert levenshtein_distance("a", "a") == 0

    def test_different_length_strings(self):
        assert levenshtein_distance("kitten", "sitting") == 3


class TestFindNearestPattern:

    def test_exact_match_returns_zero_distance(self):
        known = ["abc", "def", "ghi"]
        pattern, dist = find_nearest_pattern("abc", known)
        assert pattern == "abc"
        assert dist == 0

    def test_finds_closest_with_one_diff(self):
        known = ["aaa", "bbb", "ccc"]
        pattern, dist = find_nearest_pattern("aab", known)
        assert pattern == "aaa"
        assert dist == 1

    def test_single_known_pattern(self):
        known = ["xyz"]
        pattern, dist = find_nearest_pattern("abc", known)
        assert pattern == "xyz"
        assert dist == 3

    def test_empty_known_patterns_raises(self):
        with pytest.raises(ValueError):
            find_nearest_pattern("abc", [])

    def test_unseen_pattern_example_from_spec(self):
        # PDF örneği: "adc" -> nearest is "abc" with distance=1
        known = ["aab", "abc", "bcc"]
        pattern, dist = find_nearest_pattern("adc", known)
        assert pattern == "abc"
        assert dist == 1

    def test_returns_minimum_distance(self):
        known = ["aaaa", "abcd", "aaab"]
        # "aabb" vs "aaaa"→2, "abcd"→2, "aaab"→1
        pattern, dist = find_nearest_pattern("aabb", known)
        assert dist == 1
        assert pattern == "aaab"

    def test_sax_alphabet_patterns(self):
        known = ["aab", "abb", "bbb"]
        pattern, dist = find_nearest_pattern("aaa", known)
        assert dist == 1
        assert pattern == "aab"
