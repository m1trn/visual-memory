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


def test_a_click_in_a_resized_window_names_the_right_pixel() -> None:
    """A resizable window scales the image but reports window pixels, so a
    click has to be unscaled before it can pick out an object."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from live_demo import window_to_frame

    frame = (480, 640)                       # h, w

    # same size: unchanged
    assert window_to_frame((100, 50), (640, 480), frame) == (100, 50)

    # doubled: a click at twice the coordinates is the same pixel
    assert window_to_frame((200, 100), (1280, 960), frame) == (100, 50)

    # halved
    assert window_to_frame((50, 25), (320, 240), frame) == (100, 50)

    # wider than the aspect ratio: the image is centred, so padding comes off
    x, y = window_to_frame((320 + 100, 50), (1280, 480), frame)
    assert (x, y) == (100, 50)

    # a click outside the image is clamped into it rather than indexing wildly
    x, y = window_to_frame((5000, 5000), (640, 480), frame)
    assert 0 <= x < 640 and 0 <= y < 480
