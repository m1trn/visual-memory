"""The semantic index: what it returns, and what it refuses to return."""

from __future__ import annotations

import numpy as np
import pytest

from vision_memory.language import LanguageConfig, SemanticIndex

DIM = 16


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _direction(i: int) -> np.ndarray:
    v = np.zeros(DIM, np.float32)
    v[i] = 1.0
    return v


def test_finds_the_identity_pointing_at_the_query() -> None:
    index = SemanticIndex(DIM)
    for identity_id, axis in ((10, 0), (11, 1), (12, 2)):
        index.describe(identity_id, _direction(axis))

    hits = index.find(_direction(1), k=3, min_score=0.5)
    assert hits and hits[0][0] == 11
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)


def test_weak_matches_are_dropped_rather_than_ranked() -> None:
    """A vector index always returns its k nearest rows; the floor is what
    lets the system say nothing here matches."""
    index = SemanticIndex(DIM)
    index.describe(1, _direction(0))
    index.describe(2, _direction(1))

    query = _direction(5)  # points at nothing stored
    assert index.find(query, k=5, min_score=0.0), "without a floor, junk still ranks"
    assert index.find(query, k=5, min_score=0.2) == []


def test_describing_an_identity_twice_replaces_it() -> None:
    """An identity has one description; re-describing must not accumulate rows."""
    index = SemanticIndex(DIM)
    index.describe(7, _direction(0))
    index.describe(7, _direction(1))
    assert len(index) == 1
    hits = index.find(_direction(1), k=5, min_score=0.5)
    assert [i for i, _ in hits] == [7]
    assert index.find(_direction(0), k=5, min_score=0.5) == []


def test_forgetting_removes_it_from_search() -> None:
    index = SemanticIndex(DIM)
    index.describe(3, _direction(0))
    index.describe(4, _direction(0))
    index.forget(3)
    assert len(index) == 1
    assert [i for i, _ in index.find(_direction(0), k=5, min_score=0.5)] == [4]


def test_empty_index_answers_nothing_rather_than_failing() -> None:
    assert SemanticIndex(DIM).find(_direction(0), k=5, min_score=0.0) == []


def test_config_floor_is_on_the_text_image_scale() -> None:
    """Guards a real trap: image-image cosine runs near 0.8, text-image near
    0.28, so a threshold copied from the identity path would reject everything."""
    assert 0.1 < LanguageConfig().min_score < 0.5


def test_typing_a_query_keystroke_by_keystroke() -> None:
    """The path a user actually takes: press /, type, press enter.

    Worth pinning because while typing, `1` and `q` must be letters rather than
    the mode and quit commands - otherwise "a person in a red jacket" is
    untypable.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from live_demo import type_key

    query, typing, go = "", True, False
    for ch in "a person in a red jacket 1q":
        query, typing, go = type_key(ord(ch), query)
        assert typing and not go
    assert query == "a person in a red jacket 1q"

    query, typing, go = type_key(8, query)          # backspace
    assert query == "a person in a red jacket 1" and typing and not go

    query, typing, go = type_key(13, query)         # enter runs it
    assert not typing and go and query == "a person in a red jacket 1"

    query, typing, go = type_key(27, "half typed")  # escape abandons it
    assert query == "" and not typing and not go
