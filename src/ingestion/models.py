"""
Core data models shared across all pipeline stages.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Turn:
    turn_index: int
    speaker: str                        # "USER" or "ASSISTANT"
    text: str
    slots: list[str] = field(default_factory=list)

    # Set by landmark detector
    is_landmark: bool = False
    landmark_type: str | None = None
    landmark_reason: str = ""
    promoted: bool = False

    # Set by relevance scorer
    score: float = 0.0

    # Set by compressor
    disposition: str = ""               # "KEEP" | "CANDIDATE" | "COMPRESS"


@dataclass
class Conversation:
    conversation_id: str
    instruction_id: str
    turns: list[Turn]
    domain: str = "flights"


@dataclass
class OptimizerConfig:
    # Landmark detection
    landmark_detector: str = "rules"

    # Compression strategy
    # "turn"          — v1: threshold-based turn-level
    # "sentence"      — v2: sentence-level within landmark turns
    # "topk"          — v3: proportional top-K turn retrieval
    # "topk-sentence" — v4: top-K across all units including landmark sentences
    # "chunk"         — v5: overlapping multi-turn chunk scoring + top-K
    compression_strategy: str = "turn"

    # Query classification
    query_classifier: str = "rules"

    # Scoring weights per query type
    lambda_decay: float = 0.05
    landmark_boost: float = 0.3
    weights: dict = field(default_factory=lambda: {
        "factual":    {"keyword": 0.3, "semantic": 0.5, "recency": 0.2},
        "analytical": {"keyword": 0.2, "semantic": 0.4, "recency": 0.4},
        "preference": {"keyword": 0.2, "semantic": 0.3, "recency": 0.5},
    })

    # v1 turn-level thresholds
    thresholds: dict = field(default_factory=lambda: {
        "factual":    {"high": 0.72, "low": 0.45},
        "analytical": {"high": 0.65, "low": 0.40},
        "preference": {"high": 0.60, "low": 0.35},
    })

    # v2 sentence-level thresholds
    sentence_thresholds: dict = field(default_factory=lambda: {
        "factual":    {"high": 0.80, "low": 0.60},
        "analytical": {"high": 0.75, "low": 0.55},
        "preference": {"high": 0.70, "low": 0.50},
    })

    # v3 top-K fraction — proportion of turns to keep (landmarks compete)
    topk_fraction: dict = field(default_factory=lambda: {
        "factual":    0.20,
        "analytical": 0.35,
        "preference": 0.25,
    })
    topk_min_score: float = 0.30

    # v4 top-K sentence fraction
    topk_sentence_fraction: dict = field(default_factory=lambda: {
        "factual":    0.40,
        "analytical": 0.60,
        "preference": 0.50,
    })

    # v5 chunk-based retrieval settings.
    # chunk_size:   number of turns per chunk (rich enough for embedding)
    # chunk_stride: step between chunk start positions (overlap = size - stride)
    # chunk_topk_fraction: proportion of turns to keep after chunk scoring
    # A turn's score = max score of any chunk it appears in.
    # Landmarks receive a boost but are not hard-KEEPed.
    chunk_size:   int = 6
    chunk_stride: int = 2
    chunk_topk_fraction: dict = field(default_factory=lambda: {
        "factual":    0.35,
        "analytical": 0.55,
        "preference": 0.45,
    })

    # Model names
    embedding_model: str = "all-MiniLM-L6-v2"
    summarisation_model: str = "gpt-4o-mini"
    generator_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o"

    # Corpus
    min_turns: int = 20
    data_path: str = "data/taskmaster2/flights.json"
