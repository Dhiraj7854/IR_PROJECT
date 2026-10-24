"""Optional query transformations that preserve the user's original query."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def load_synonyms(path: Path) -> Dict[str, list[str]]:
    """Load an editable JSON mapping from terms to expansion phrases."""
    if not path.is_file():
        raise FileNotFoundError(f"Synonym dictionary not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Synonym dictionary must be a JSON object")
    synonyms: Dict[str, list[str]] = {}
    for term, expansions in raw.items():
        if not isinstance(term, str) or not term.strip():
            raise ValueError("Synonym keys must be non-empty strings")
        if isinstance(expansions, str):
            expansions = [expansions]
        if not isinstance(expansions, list) or any(
            not isinstance(item, str) or not item.strip() for item in expansions
        ):
            raise ValueError(f"Synonyms for {term!r} must be non-empty strings")
        synonyms[term.casefold()] = [item.strip() for item in expansions]
    return synonyms


def enhance_query(
    query: str,
    method: str = "original",
    *,
    synonyms: Optional[Mapping[str, list[str]]] = None,
    spellchecker: Any = None,
) -> str:
    """Transform a query with opt-in normalization, spelling, or synonym expansion."""
    original = query or ""
    if method == "original":
        return original
    if method not in {"normalize", "spell", "synonym"}:
        raise ValueError(f"Unknown query enhancement method: {method!r}")
    normalized = unicodedata.normalize("NFKC", original)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if method == "normalize":
        return normalized.casefold()
    if method == "synonym":
        if not normalized:
            return ""
        dictionary = {key.casefold(): value for key, value in (synonyms or {}).items()}
        tokens = normalized.split()
        expansions = []
        seen = {token.casefold() for token in tokens}
        for token in tokens:
            for expansion in dictionary.get(token.casefold(), []):
                key = expansion.casefold()
                if key not in seen:
                    expansions.append(expansion)
                    seen.add(key)
        return " ".join([normalized, *expansions])

    if not normalized:
        return ""
    if spellchecker is None:
        try:
            from spellchecker import SpellChecker
        except ImportError as error:
            raise RuntimeError(
                "Spelling correction requires the optional 'pyspellchecker' package"
            ) from error
        spellchecker = SpellChecker()

    protected = set((synonyms or {}).keys())
    tokens = re.findall(r"\w+|[^\w\s]+|\s+", normalized, flags=re.UNICODE)
    corrected = []
    for token in tokens:
        if (
            not token.isalpha()
            or len(token) < 4
            or token[0].isupper()
            or token.casefold() in protected
        ):
            corrected.append(token)
            continue
        correction = spellchecker.correction(token)
        corrected.append(correction or token)
    return "".join(corrected)
