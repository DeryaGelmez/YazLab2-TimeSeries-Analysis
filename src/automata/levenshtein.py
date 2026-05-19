def levenshtein_distance(str1, str2):

    len1 = len(str1)
    len2 = len(str2)

    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        dp[i][0] = i

    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):

            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # substitution
            )

    return dp[len1][len2]


def find_nearest_pattern(unseen_pattern, known_patterns):

    if not known_patterns:
        raise ValueError("known_patterns cannot be empty.")

    nearest_pattern = None
    min_distance = float("inf")

    for pattern in known_patterns:
        distance = levenshtein_distance(unseen_pattern, pattern)

        if distance < min_distance:
            min_distance = distance
            nearest_pattern = pattern

    return nearest_pattern, min_distance