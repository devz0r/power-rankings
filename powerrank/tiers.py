"""Tier list: split the ratings at their natural gaps.

Tiers come from Fisher-Jenks natural breaks, which choose the split that keeps
each tier as tight as possible, so lines fall where the gaps between teams
are largest instead of cutting the league into equal-sized groups. Any one
team's rating is uncertain by a few points, so a team near a line could
belong on either side of it; the ranked view shows each team's range.
"""

from __future__ import annotations

TIERS = ("S", "A", "B", "C", "D", "E", "F")


def natural_breaks(values: list[float], k: int) -> list[int]:
    """Group index (0 = highest) for each value, using Fisher-Jenks."""
    order = sorted(range(len(values)), key=lambda i: -values[i])
    xs = [values[i] for i in order]
    n = len(xs)
    k = min(k, n)
    prefix = [0.0]
    prefix_sq = [0.0]
    for v in xs:
        prefix.append(prefix[-1] + v)
        prefix_sq.append(prefix_sq[-1] + v * v)

    def cost(i: int, j: int) -> float:
        """Sum of squared deviations of xs[i:j]."""
        s = prefix[j] - prefix[i]
        return prefix_sq[j] - prefix_sq[i] - s * s / (j - i)

    inf = float("inf")
    best = [[inf] * (n + 1) for _ in range(k + 1)]
    split = [[0] * (n + 1) for _ in range(k + 1)]
    best[0][0] = 0.0
    for groups in range(1, k + 1):
        for end in range(groups, n + 1):
            for start in range(groups - 1, end):
                c = best[groups - 1][start] + cost(start, end)
                if c < best[groups][end]:
                    best[groups][end] = c
                    split[groups][end] = start
    labels = [0] * n
    end = n
    for groups in range(k, 0, -1):
        start = split[groups][end]
        for pos in range(start, end):
            labels[order[pos]] = groups - 1
        end = start
    return labels


def assign(ratings: dict[str, float]) -> dict[str, str]:
    teams = list(ratings)
    labels = natural_breaks([ratings[t] for t in teams], len(TIERS))
    return {t: TIERS[label] for t, label in zip(teams, labels)}
