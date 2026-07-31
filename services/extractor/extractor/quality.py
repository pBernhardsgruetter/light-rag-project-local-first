"""Small, dependency-free helpers shared by extraction and retrieval.

Keeping these rules in one place prevents the extractor and retriever from
normalizing names and relation labels differently.
"""

import re
import unicodedata
from typing import Iterable, List, Optional, Sequence, Tuple


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def normalize_text(value: str) -> str:
    """Return a stable, case-insensitive form suitable for matching."""
    value = unicodedata.normalize("NFKC", value or "").casefold()
    return " ".join(_TOKEN_RE.findall(value))


def tokens(value: str) -> List[str]:
    return normalize_text(value).split()


def lexical_score(query: str, text: str) -> float:
    """Score token overlap with a small exact-phrase bonus."""
    query_norm = normalize_text(query)
    text_norm = normalize_text(text)
    if not query_norm or not text_norm:
        return 0.0

    query_tokens = set(query_norm.split())
    text_tokens = set(text_norm.split())
    overlap = len(query_tokens & text_tokens) / max(len(query_tokens), 1)
    phrase_bonus = 0.25 if query_norm in text_norm else 0.0
    return min(1.0, overlap + phrase_bonus)


def normalize_relation_type(value: str, allowed: Optional[Iterable[str]] = None) -> Tuple[str, str]:
    """Map common relation phrasing to a stable predicate.

    The raw normalized label is returned as the second item so evidence can
    retain what the relation model actually emitted.
    """
    raw = "_".join(_TOKEN_RE.findall((value or "RELATED_TO").upper()))
    aliases = {
        "EMPLOYED_BY": "WORKS_FOR",
        "WORKS_FOR": "WORKS_FOR",
        "WORKS_ON": "WORKS_ON",
        "LOCATED_AT": "LOCATED_IN",
        "LOCATED_IN": "LOCATED_IN",
        "PARTICIPATES_IN": "PARTICIPATED_IN",
        "PARTICIPATED_IN": "PARTICIPATED_IN",
        "AFFECTS": "IMPACTED",
        "IMPACTS": "IMPACTED",
        "IMPACTED": "IMPACTED",
        "CREATED": "DEVELOPED",
        "DEVELOPS": "DEVELOPED",
        "DEVELOPED": "DEVELOPED",
        "USES": "USES",
        "REGULATES": "REGULATES",
        "PARTNERS_WITH": "PARTNERED_WITH",
        "PARTNERED_WITH": "PARTNERED_WITH",
    }
    predicate = aliases.get(raw, raw or "RELATED_TO")
    allowed_set = {str(item).upper() for item in allowed or []}
    if allowed_set and predicate not in allowed_set:
        predicate = "RELATED_TO"
    return predicate, raw or "RELATED_TO"


def resolve_endpoint(value: str, candidates: Sequence[Tuple[str, str]]) -> Optional[str]:
    """Resolve a relation endpoint against (surface, entity_id) candidates.

    Exact normalized matches win. Substring matching is only accepted when it
    produces one unambiguous longest candidate, avoiding arbitrary matches.
    """
    needle = normalize_text(value)
    if not needle:
        return None

    exact = [entity_id for surface, entity_id in candidates if normalize_text(surface) == needle]
    if len(set(exact)) == 1:
        return exact[0]
    if exact:
        return None

    possible = [
        (len(normalize_text(surface)), entity_id)
        for surface, entity_id in candidates
        if normalize_text(surface) in needle or needle in normalize_text(surface)
    ]
    if not possible:
        return None
    possible.sort(reverse=True)
    longest = [entity_id for length, entity_id in possible if length == possible[0][0]]
    return longest[0] if len(set(longest)) == 1 else None
