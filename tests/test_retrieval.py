"""Rank-N and mAP: the protocol must not be able to flatter itself."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from reid_retrieval import evaluate  # noqa: E402

DIM = 8


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _person(axis, n, jitter, rng):
    base = np.zeros(DIM, np.float32)
    base[axis] = 1.0
    return _unit(base + rng.normal(scale=jitter, size=(n, DIM)).astype(np.float32))


def test_a_perfectly_separated_gallery_scores_one() -> None:
    rng = np.random.default_rng(0)
    vectors = np.concatenate([_person(0, 4, 0.01, rng), _person(1, 4, 0.01, rng)])
    people = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    tracks = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    r = evaluate(vectors, people, tracks)
    assert r["queries"] == 8
    assert r["rank1"] == 1.0 and r["map"] == 1.0


def test_same_track_neighbours_cannot_answer_the_query() -> None:
    """Adjacent frames of one track are the same observation twice; counting
    them would report a high score for an embedder that had learnt nothing."""
    rng = np.random.default_rng(1)
    # Two people, each seen on ONE track only: no honest answer exists.
    vectors = np.concatenate([_person(0, 3, 0.01, rng), _person(1, 3, 0.01, rng)])
    people = np.array([1, 1, 1, 2, 2, 2])
    tracks = np.array([1, 1, 1, 2, 2, 2])
    assert evaluate(vectors, people, tracks)["queries"] == 0


def test_a_useless_embedder_scores_near_chance() -> None:
    rng = np.random.default_rng(2)
    vectors = _unit(rng.normal(size=(40, DIM)).astype(np.float32))
    people = np.array([i % 10 for i in range(40)])
    tracks = np.arange(40)
    r = evaluate(vectors, people, tracks)
    assert r["rank1"] < 0.4, r["rank1"]


def test_ranks_are_monotone_and_map_never_exceeds_rank1_ceiling() -> None:
    rng = np.random.default_rng(3)
    vectors = np.concatenate([_person(i % 4, 3, 0.35, rng) for i in range(8)])
    people = np.repeat(np.arange(8) % 4, 3)
    tracks = np.repeat(np.arange(8), 3)
    r = evaluate(vectors, people, tracks)
    assert r["rank1"] <= r["rank5"] <= r["rank10"]
    assert 0.0 <= r["map"] <= 1.0
