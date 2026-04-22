"""
Semantic similarity scorer using local sentence-transformers embeddings.

Uses all-MiniLM-L6-v2 — loaded once at module level, cached per conversation.
Zero API cost. Deterministic. ~80MB model.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL: SentenceTransformer | None = None


def _get_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(model_name)
    return _MODEL


def semantic_scores(
    query: str,
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> list[float]:
    """
    Compute cosine similarity between the query embedding and each turn embedding.

    Turn embeddings are computed in a single batch call for efficiency.
    The model is loaded once and cached for the process lifetime.

    Args:
        query:      Current query string.
        texts:      Turn texts (history only — turns 0..Q-1).
        model_name: Sentence-transformers model to use.

    Returns:
        List of floats in [0, 1], one per turn.
    """
    if not texts:
        return []

    model = _get_model(model_name)

    # Encode query and all turns in one batch
    all_texts = texts + [query]
    embeddings = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=False)

    turn_embeddings = embeddings[:-1]   # shape: (N, dim)
    query_embedding = embeddings[-1]    # shape: (dim,)

    # Cosine similarity: dot product of normalised vectors
    turn_norms  = np.linalg.norm(turn_embeddings, axis=1, keepdims=True)
    query_norm  = np.linalg.norm(query_embedding)

    # Avoid division by zero
    turn_norms  = np.where(turn_norms == 0, 1e-9, turn_norms)
    query_norm  = query_norm if query_norm > 0 else 1e-9

    turn_normed  = turn_embeddings / turn_norms
    query_normed = query_embedding / query_norm

    scores = (turn_normed @ query_normed).tolist()

    # Clip to [0, 1] — cosine can be negative for dissimilar texts
    return [max(0.0, float(s)) for s in scores]
