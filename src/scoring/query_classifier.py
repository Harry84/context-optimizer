"""
Query classifier — classifies a query as factual, analytical, or preference.
Drives weight and threshold selection for relevance scoring.
"""

from __future__ import annotations
import re
from typing import Literal

QueryType = Literal["factual", "analytical", "preference"]

_ANALYTICAL = [re.compile(p, re.IGNORECASE) for p in [
    r"\b(why|compare|explain|difference|reason|better|worse|pros|cons)\b",
    r"\bhow (do|does|did|would|could|can)\b",
]]
_PREFERENCE = [re.compile(p, re.IGNORECASE) for p in [
    r"\b(recommend|suggest|best|prefer|which would|which should|what would you)\b",
]]
_FACTUAL = [re.compile(p, re.IGNORECASE) for p in [
    r"\b(what|when|where|who|which|how much|how many|how long)\b",
    r"\b(price|cost|fare|date|time|airport|airline|stop|layover|seat)\b",
]]


def classify_query(query: str) -> QueryType:
    """
    Classify a query string into one of three types.

    Order of precedence: analytical > preference > factual > analytical (default).
    Analytical is the safe default — it preserves the most context.
    """
    q = query.lower()
    if any(p.search(q) for p in _ANALYTICAL):  return "analytical"
    if any(p.search(q) for p in _PREFERENCE):  return "preference"
    if any(p.search(q) for p in _FACTUAL):     return "factual"
    return "analytical"
