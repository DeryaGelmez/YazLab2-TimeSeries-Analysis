import pytest
from src.automata.transition import (
    count_transitions,
    calculate_transition_probabilities,
    calculate_path_probability,
    calculate_transition_density,
)


class TestCountTransitions:

    def test_basic_sequence(self):
        patterns = ["a", "b", "a", "c"]
        counts = count_transitions(patterns)
        assert counts["a"]["b"] == 1
        assert counts["a"]["c"] == 1
        assert counts["b"]["a"] == 1

    def test_repeated_transition(self):
        patterns = ["a", "b", "a", "b", "a", "b"]
        counts = count_transitions(patterns)
        assert counts["a"]["b"] == 3
        assert counts["b"]["a"] == 2

    def test_single_pattern_raises(self):
        with pytest.raises(ValueError):
            count_transitions(["a"])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            count_transitions([])

    def test_two_patterns_one_transition(self):
        counts = count_transitions(["x", "y"])
        assert counts["x"]["y"] == 1

    def test_self_transition(self):
        patterns = ["a", "a", "a"]
        counts = count_transitions(patterns)
        assert counts["a"]["a"] == 2


class TestCalculateTransitionProbabilities:

    def test_probabilities_sum_to_one_per_state(self):
        patterns = ["a", "b", "a", "c", "a", "b"]
        counts = count_transitions(patterns)
        probs = calculate_transition_probabilities(counts)
        total = sum(probs["a"].values())
        assert abs(total - 1.0) < 1e-9

    def test_single_outgoing_transition_is_one(self):
        patterns = ["a", "b", "a", "b"]
        counts = count_transitions(patterns)
        probs = calculate_transition_probabilities(counts)
        assert probs["a"]["b"] == 1.0

    def test_two_equal_transitions(self):
        patterns = ["a", "b", "a", "c"]
        counts = count_transitions(patterns)
        probs = calculate_transition_probabilities(counts)
        assert abs(probs["a"]["b"] - 0.5) < 1e-9
        assert abs(probs["a"]["c"] - 0.5) < 1e-9

    def test_known_proportions(self):
        # a->b occurs 3 times, a->c occurs 1 time → P(a->b)=0.75, P(a->c)=0.25
        patterns = ["a", "b", "a", "b", "a", "b", "a", "c"]
        counts = count_transitions(patterns)
        probs = calculate_transition_probabilities(counts)
        assert abs(probs["a"]["b"] - 0.75) < 1e-9
        assert abs(probs["a"]["c"] - 0.25) < 1e-9


class TestCalculatePathProbability:

    def test_single_transition_probability(self):
        probs = {"a": {"b": 0.8}, "b": {"c": 0.5}}
        result = calculate_path_probability(["a", "b"], probs)
        assert abs(result - 0.8) < 1e-9

    def test_two_transitions_multiplied(self):
        probs = {"a": {"b": 0.8}, "b": {"c": 0.5}}
        result = calculate_path_probability(["a", "b", "c"], probs)
        assert abs(result - 0.4) < 1e-9

    def test_missing_transition_returns_zero(self):
        probs = {"a": {"b": 0.8}}
        result = calculate_path_probability(["a", "x"], probs)
        assert result == 0.0

    def test_single_pattern_returns_one(self):
        probs = {"a": {"b": 0.8}}
        result = calculate_path_probability(["a"], probs)
        assert result == 1.0

    def test_empty_patterns_returns_one(self):
        probs = {}
        result = calculate_path_probability([], probs)
        assert result == 1.0

    def test_unknown_state_returns_zero(self):
        probs = {"a": {"b": 0.9}}
        result = calculate_path_probability(["z", "b"], probs)
        assert result == 0.0

    def test_three_state_path(self):
        probs = {
            "a": {"b": 0.72},
            "b": {"c": 0.15},
        }
        result = calculate_path_probability(["a", "b", "c"], probs)
        assert abs(result - 0.108) < 1e-9


class TestCalculateTransitionDensity:

    def test_fully_connected_two_states(self):
        patterns = ["a", "b", "a", "b"]
        counts = count_transitions(patterns)
        states = {"a", "b"}
        density = calculate_transition_density(counts, states)
        # observed: a->b, b->a = 2 transitions, possible: 2*2=4
        assert abs(density - 0.5) < 1e-9

    def test_empty_states_returns_zero(self):
        from collections import defaultdict
        counts = defaultdict(lambda: defaultdict(int))
        density = calculate_transition_density(counts, set())
        assert density == 0.0

    def test_density_between_zero_and_one(self):
        patterns = ["a", "b", "c", "a", "b"]
        counts = count_transitions(patterns)
        states = {"a", "b", "c"}
        density = calculate_transition_density(counts, states)
        assert 0.0 <= density <= 1.0
