from collections import defaultdict


def count_transitions(patterns):

    if len(patterns) < 2:
        raise ValueError("At least two patterns are required to count transitions.")

    transition_counts = defaultdict(lambda: defaultdict(int))

    for current_state, next_state in zip(patterns[:-1], patterns[1:]):
        transition_counts[current_state][next_state] += 1

    return transition_counts


def calculate_transition_probabilities(transition_counts):

    transition_probabilities = {}

    for current_state, next_states in transition_counts.items():
        total_outgoing = sum(next_states.values())

        transition_probabilities[current_state] = {}

        for next_state, count in next_states.items():
            probability = count / total_outgoing
            transition_probabilities[current_state][next_state] = probability

    return transition_probabilities


def calculate_path_probability(patterns, transition_probabilities):
  
    if len(patterns) < 2:
        return 1.0

    path_probability = 1.0

    for current_state, next_state in zip(patterns[:-1], patterns[1:]):

        probability = transition_probabilities.get(
            current_state,
            {}
        ).get(
            next_state,
            0.0
        )

        path_probability *= probability

    return path_probability


def calculate_transition_density(transition_counts, states):

    observed_transitions = sum(
        len(next_states)
        for next_states in transition_counts.values()
    )

    possible_transitions = len(states) * len(states)

    if possible_transitions == 0:
        return 0.0

    return observed_transitions / possible_transitions