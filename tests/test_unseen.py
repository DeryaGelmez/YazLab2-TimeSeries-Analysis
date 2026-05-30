"""
Unseen pattern yönetimi birim testleri.

PDF gereksinimi: "Bu mekanizmanın birim testlerle doğrulanması zorunludur."
Test edilen mekanizma: Levenshtein tabanlı en yakın pattern eşleme ve
kontrollü unseen pattern oluşturma.
"""
import random
import pytest
from src.automata.levenshtein import levenshtein_distance, find_nearest_pattern
from src.automata.pattern import extract_patterns, get_unique_states
from src.automata.transition import (
    count_transitions,
    calculate_transition_probabilities,
    calculate_path_probability,
)


def create_unseen_pattern(pattern, alphabet_size, seed=None):
    """Test patterns için deterministik unseen oluşturucu."""
    if seed is not None:
        random.seed(seed)
    alphabet = [chr(i) for i in range(97, 97 + alphabet_size)]
    chars = list(pattern)
    last = chars[-1]
    choices = [c for c in alphabet if c != last]
    chars[-1] = random.choice(choices)
    return "".join(chars)


class TestUnseenPatternCreation:

    def test_created_pattern_differs_from_original(self):
        original = "abc"
        new_pattern = create_unseen_pattern(original, alphabet_size=3, seed=42)
        assert new_pattern != original

    def test_created_pattern_same_length(self):
        original = "abcd"
        new_pattern = create_unseen_pattern(original, alphabet_size=4, seed=0)
        assert len(new_pattern) == len(original)

    def test_only_last_char_modified(self):
        original = "abc"
        new_pattern = create_unseen_pattern(original, alphabet_size=3, seed=1)
        assert new_pattern[:2] == original[:2]

    def test_last_char_is_in_alphabet(self):
        original = "abc"
        alphabet = set("abc")
        new_pattern = create_unseen_pattern(original, alphabet_size=3, seed=7)
        assert new_pattern[-1] in alphabet

    def test_last_char_differs_from_original_last(self):
        original = "abc"
        for seed in range(20):
            new_pattern = create_unseen_pattern(original, alphabet_size=3, seed=seed)
            assert new_pattern[-1] != original[-1]


class TestUnseenMappingMechanism:

    def test_unseen_maps_to_nearest_known(self):
        known = {"aab", "abc", "bcc"}
        unseen = "adc"
        nearest, dist = find_nearest_pattern(unseen, list(known))
        assert nearest == "abc"
        assert dist == 1

    def test_unseen_mapping_distance_is_minimal(self):
        known = ["aaaa", "bbbb", "cccc", "aabb"]
        unseen = "aabc"
        nearest, dist = find_nearest_pattern(unseen, known)
        for p in known:
            assert levenshtein_distance(unseen, p) >= dist

    def test_seen_pattern_maps_to_itself(self):
        known = {"abc", "bcd", "cde"}
        seen = "abc"
        nearest, dist = find_nearest_pattern(seen, list(known))
        assert nearest == seen
        assert dist == 0

    def test_mapping_does_not_alter_known_states(self):
        known = {"aab", "abc", "bcc"}
        known_copy = known.copy()
        find_nearest_pattern("xyz", list(known))
        assert known == known_copy


class TestUnseenInPipeline:
    """Otomata pipeline'ında unseen mekanizmasının entegrasyon testleri."""

    def setup_method(self):
        train_seq = "aabbbcccaaabbbccc"
        self.train_patterns = extract_patterns(train_seq, window_size=3)
        self.train_states = get_unique_states(self.train_patterns)
        counts = count_transitions(self.train_patterns)
        self.probs = calculate_transition_probabilities(counts)

    def test_unseen_pattern_not_in_train_states(self):
        all_patterns = ["aab", "abb", "bbb", "bbc", "bcc", "ccc", "cca", "caa", "aaa"]
        unseen = "zzz"
        assert unseen not in self.train_states

    def test_pipeline_handles_unseen_via_levenshtein(self):
        unseen = "zzz"
        assert unseen not in self.train_states
        nearest, dist = find_nearest_pattern(unseen, list(self.train_states))
        assert nearest in self.train_states
        assert dist >= 0

    def test_path_probability_of_unseen_mapped_pattern(self):
        unseen = "zzz"
        nearest, _ = find_nearest_pattern(unseen, list(self.train_states))
        # Path probability should be calculable after mapping
        if self.train_patterns:
            next_p = self.train_patterns[0]
            if next_p not in self.train_states:
                next_p, _ = find_nearest_pattern(next_p, list(self.train_states))
            prob = calculate_path_probability([nearest, next_p], self.probs)
            assert isinstance(prob, float)
            assert 0.0 <= prob <= 1.0

    def test_zero_probability_triggers_anomaly_decision(self):
        # A path with zero probability → anomaly (threshold > 0)
        probs = {"a": {"b": 0.9}}
        prob = calculate_path_probability(["a", "x"], probs)
        threshold = 0.10
        prediction = 1 if prob < threshold else 0
        assert prediction == 1

    def test_high_probability_triggers_normal_decision(self):
        probs = {"a": {"b": 0.9}}
        prob = calculate_path_probability(["a", "b"], probs)
        threshold = 0.10
        prediction = 1 if prob < threshold else 0
        assert prediction == 0
