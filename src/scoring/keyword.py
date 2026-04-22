"""
TF-IDF keyword match scorer.

Fits a TF-IDF vectoriser on the conversation history (turns 0..Q-1)
so IDF reflects this conversation's vocabulary, then computes cosine
similarity between the query vector and each turn vector.
"""

from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def keyword_scores(query: str, texts: list[str]) -> list[float]:
    """
    Compute TF-IDF cosine similarity between query and each turn text.

    Args:
        query: The current query string.
        texts: List of turn texts (history only — turns 0..Q-1).

    Returns:
        List of floats in [0, 1], one per turn.
    """
    if not texts:
        return []

    corpus = texts + [query]
    try:
        vectoriser = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )
        tfidf = vectoriser.fit_transform(corpus)
    except ValueError:
        # Empty vocabulary — all texts are stop words or empty
        return [0.0] * len(texts)

    turn_vecs  = tfidf[:-1]   # all rows except last
    query_vec  = tfidf[-1]    # last row

    scores = cosine_similarity(query_vec, turn_vecs).flatten()
    return scores.tolist()
