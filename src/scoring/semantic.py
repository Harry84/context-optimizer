"""
Semantic similarity scorer using local sentence-transformers embeddings.
"""

from __future__ import annotations

import logging
import numpy as np

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer

_MODEL: SentenceTransformer | None = None


def _get_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(model_name, tokenizer_kwargs={"clean_up_tokenization_spaces": True})
    return _MODEL


def semantic_scores(
    query: str,
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> list[float]:
    """
    Compute cosine similarity between the query embedding and each turn embedding.
    """
    if not texts:
        return []

    model = _get_model(model_name)
    all_texts  = texts + [query]
    embeddings = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=False)

    turn_embeddings = embeddings[:-1]
    query_embedding = embeddings[-1]

    turn_norms  = np.linalg.norm(turn_embeddings, axis=1, keepdims=True)
    query_norm  = np.linalg.norm(query_embedding)

    turn_norms  = np.where(turn_norms == 0, 1e-9, turn_norms)
    query_norm  = query_norm if query_norm > 0 else 1e-9

    turn_normed  = turn_embeddings / turn_norms
    query_normed = query_embedding / query_norm

    scores = (turn_normed @ query_normed).tolist()
    return [max(0.0, float(s)) for s in scores]
