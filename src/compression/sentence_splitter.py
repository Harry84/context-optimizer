"""
Sentence boundary splitter using NLTK's punkt tokenizer directly.

Provides a clean split_sentences(text) -> list[str] interface used by
sentence_compressor.py. LangChain is in the dependency tree for the
embedding work planned in v2 (KD-011); NLTK punkt is used here for
accurate sentence boundary detection.

Requires: nltk and punkt_tab data.
Download once with: python -c "import nltk; nltk.download('punkt_tab')"
"""

from __future__ import annotations

import nltk


def split_sentences(text: str) -> list[str]:
    """
    Split text into a list of sentences using NLTK's punkt tokenizer.

    Returns a list of non-empty stripped strings. Always returns at least
    one element (the original text) if no boundaries are detected.
    """
    if not text or not text.strip():
        return []

    sentences = nltk.sent_tokenize(text.strip())
    cleaned = [s.strip() for s in sentences if s.strip()]
    return cleaned if cleaned else [text.strip()]
