"""
Recency decay scorer.

Assigns higher scores to turns closer to the current query position.
Uses exponential decay: exp(-lambda * (Q - turn_index))

Where Q is the query position (index of the current query turn)
and turn_index is the position of the history turn being scored.
"""

from __future__ import annotations

import math


def recency_scores(
    turn_indices: list[int],
    query_position: int,
    lambda_decay: float = 0.05,
) -> list[float]:
    """
    Compute recency decay scores for a list of turn indices.

    Args:
        turn_indices:   List of turn_index values for history turns (0..Q-1).
        query_position: Index Q of the current query turn.
        lambda_decay:   Decay rate. Higher = faster decay. Default 0.05.
                        At lambda=0.05, a turn 20 positions back scores ~0.37.

    Returns:
        List of floats in (0, 1], one per turn. Most recent turn scores highest.
    """
    return [
        math.exp(-lambda_decay * (query_position - idx))
        for idx in turn_indices
    ]
