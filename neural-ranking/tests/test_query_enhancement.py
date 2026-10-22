"""Tests for safe, opt-in query enhancements."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.query_enhancement import enhance_query, load_synonyms


def test_original_and_normalization_preserve_empty_input():
    assert enhance_query("  MiXeD\t query ", "original") == "  MiXeD\t query "
    assert enhance_query("  MiXeD\t query ", "normalize") == "mixed query"
    assert enhance_query("", "normalize") == ""


def test_synonym_expansion_is_opt_in_and_deduplicated():
    synonyms = {"car": ["automobile", "vehicle"], "vehicle": ["car"]}

    assert enhance_query("car", "original", synonyms=synonyms) == "car"
    assert enhance_query("car vehicle", "synonym", synonyms=synonyms) == (
        "car vehicle automobile"
    )


def test_spell_correction_protects_names_and_technical_terms():
    class StubSpellchecker:
        def correction(self, word):
            return {"recieve": "receive", "pyhton": "python"}.get(word, word)

    transformed = enhance_query(
        "recieve Pyhton API",
        "spell",
        synonyms={"pyhton": ["Python"]},
        spellchecker=StubSpellchecker(),
    )

    assert transformed == "receive Pyhton API"


def test_synonym_dictionary_rejects_invalid_json_shape(tmp_path):
    path = tmp_path / "synonyms.json"
    path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        load_synonyms(path)
