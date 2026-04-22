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
                                        # Slot names from Taskmaster-2 annotations.
                                        # EVALUATION GROUND TRUTH ONLY.
                                        # Never read by landmark detector or scorer.

    # Set by landmark detector
    is_landmark: bool = False
    landmark_type: str | None = None    # "intent" | "decision" | "action_item"
    landmark_reason: str = ""           # Human-readable reason — auditable
    promoted: bool = False              # True if promoted by cross-turn alignment

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
    landmark_detector: str = "rules"    # "rules" | "embedding" | "llm"

    # Compression strategy
    # "turn"     — v1: classify and compress whole turns (fast, safe)
    # "sentence" — v2: split landmark turns into sentences, only KEEP
    #              the landmark sentences, compress the filler around them
    compression_strategy: str = "turn"

    # Query classification
    query_classifier: str = "rules"     # "rules" | "llm"

    # Scoring weights per query type
    lambda_decay: float = 0.05
    landmark_boost: float = 0.3
    weights: dict = field(default_factory=lambda: {
        "factual":    {"keyword": 0.3, "semantic": 0.5, "recency": 0.2},
        "analytical": {"keyword": 0.2, "semantic": 0.4, "recency": 0.4},
        "preference": {"keyword": 0.2, "semantic": 0.3, "recency": 0.5},
    })

    # Turn-level compression thresholds (v1 strategy).
    thresholds: dict = field(default_factory=lambda: {
        "factual":    {"high": 0.72, "low": 0.45},
        "analytical": {"high": 0.65, "low": 0.40},
        "preference": {"high": 0.60, "low": 0.35},
    })

    # Sentence-level compression thresholds (v2 strategy).
    # Tighter than turn-level because individual sentences have less context,
    # making semantic similarity scores noisier — we need a higher bar to
    # avoid keeping weak-signal filler sentences.
    sentence_thresholds: dict = field(default_factory=lambda: {
        "factual":    {"high": 0.80, "low": 0.60},
        "analytical": {"high": 0.75, "low": 0.55},
        "preference": {"high": 0.70, "low": 0.50},
    })

    # Model names
    embedding_model: str = "all-MiniLM-L6-v2"
    summarisation_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o"

    # Corpus
    min_turns: int = 20
    data_path: str = "data/taskmaster2/flights.json"
