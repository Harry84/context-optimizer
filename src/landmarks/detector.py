"""
Landmark detector interface and factory.

Active detector selected by OptimizerConfig.landmark_detector:
  "rules"     — v1: rule-based + two-pass alignment (default)
  "embedding" — v2: prototype embedding similarity (not yet implemented)
  "llm"       — optional: LLM batch classification (not yet implemented)
"""

from __future__ import annotations
from typing import Protocol

from src.ingestion.models import Conversation, OptimizerConfig


class LandmarkDetector(Protocol):
    def detect(self, conversation: Conversation) -> Conversation:
        """
        Annotate all turns in the conversation in-place.
        Sets: is_landmark, landmark_type, landmark_reason, promoted.
        Returns the same Conversation object.
        """
        ...


def get_detector(config: OptimizerConfig) -> LandmarkDetector:
    """Factory: return the configured detector implementation."""
    if config.landmark_detector == "rules":
        from src.landmarks.rule_detector import RuleLandmarkDetector
        return RuleLandmarkDetector()
    elif config.landmark_detector == "embedding":
        raise NotImplementedError("Embedding detector is a v2 feature.")
    elif config.landmark_detector == "llm":
        raise NotImplementedError("LLM detector is an optional mode — see llm_detector.py.")
    else:
        raise ValueError(f"Unknown landmark_detector: {config.landmark_detector!r}")
