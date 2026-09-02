"""Score re-identification as retrieval: Rank-1, Rank-5 and mAP.

The rest of this project measures tracking (IDF1, identity switches), which is
what a viewer experiences. The re-identification literature measures retrieval:
given one image of a person, how far down a ranked gallery is the next image of
that same person? Market-1501 and MSMT17 report Rank-N and mean average
precision, and those are the numbers a reader from that field looks for first.

Both are worth having because they answer different questions. Rank-1 judges
the embedding alone, with no tracker, no threshold and no binding logic
involved. IDF1 judges the whole system. A high Rank-1 with a poor IDF1 means
the features are fine and the decision logic is not.

Protocol follows the standard: each observation is used in turn as a query
against a gallery of every other observation, and matches from the same track
are excluded so that adjacent frames of one track cannot trivially answer the
query. The remaining correct answers are that labelled person seen on a
DIFFERENT track - exactly what re-identification exists to recover.

Runs on the embeddings already cached by reid_sweep.py; no model is loaded.

Usage: python scripts/reid_retrieval.py [--cache data/mot/detection_cache_09.pkl]
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vision_memory.config import load_appearance_config, load_tracker_config  # noqa: E402
from vision_memory.motchallenge import load_sequence  # noqa: E402
from vision_memory.tracker import ByteTracker, iou_matrix  # noqa: E402

_ATTRIBUTION_IOU = 0.5
_MIN_VISIBILITY = 0.25


def collect(cache: Path, sequence) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Embeddings with the labelled person and the track each came from."""
    frames = pickle.loads(cache.read_bytes())
    tracker = ByteTracker(load_tracker_config())
    vectors, people, tracks = [], [], []
    for number, detections, embeddings in frames:
        active = tracker.update(detections, embeddings)
        truth = sequence.visible(number, _MIN_VISIBILITY)
        if len(truth.ids) == 0:
            continue
        fresh = [t for t in active if t.time_since_update == 0 and t.label == "person" and t.exemplars]
        if not fresh:
            continue
        overlap = iou_matrix(np.array([t.box for t in fresh], np.float64), truth.boxes)
        for row, track in enumerate(fresh):
            best = int(overlap[row].argmax())
            if overlap[row, best] < _ATTRIBUTION_IOU:
                continue
            vectors.append(np.asarray(track.exemplars[-1], np.float32))
            people.append(int(truth.ids[best]))
            tracks.append(track.id)
    return np.stack(vectors), np.array(people), np.array(tracks)


def evaluate(vectors: np.ndarray, people: np.ndarray, tracks: np.ndarray,
             ranks=(1, 5, 10)) -> dict:
    """Rank-N and mAP over all queries, excluding same-track gallery entries."""
    sims = vectors @ vectors.T
    hits = {k: 0 for k in ranks}
    average_precisions, used = [], 0
    for i in range(len(vectors)):
        # The gallery excludes the query itself and everything from its own
        # track: neighbouring frames of one track are the same observation
        # twice over, and counting them measures nothing.
        gallery = (tracks != tracks[i])
        correct = gallery & (people == people[i])
        if not correct.any():
            continue                      # this person never appears on another track
        used += 1
        order = np.argsort(-np.where(gallery, sims[i], -np.inf))
        order = order[gallery[order]]
        relevant = correct[order]
        for k in ranks:
            if relevant[:k].any():
                hits[k] += 1
        found = np.flatnonzero(relevant)
        precision = (np.arange(len(found)) + 1) / (found + 1)
        average_precisions.append(float(precision.mean()))
    return {"queries": used, "gallery": len(vectors),
            **{f"rank{k}": hits[k] / max(used, 1) for k in ranks},
            "map": float(np.mean(average_precisions)) if average_precisions else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/mot/MOT17-09-FRCNN"))
    ap.add_argument("--cache", type=Path, default=Path("data/mot/detection_cache_09.pkl"))
    args = ap.parse_args()

    sequence = load_sequence(args.data)
    vectors, people, tracks = collect(args.cache, sequence)
    result = evaluate(vectors, people, tracks)

    print(f"{sequence.name}  embedder: {load_appearance_config().model}")
    print(f"  gallery {result['gallery']} observations, {result['queries']} usable queries")
    print(f"  (a query is usable only when its person also appears on another track)")
    print()
    print(f"  Rank-1   {result['rank1']:.1%}")
    print(f"  Rank-5   {result['rank5']:.1%}")
    print(f"  Rank-10  {result['rank10']:.1%}")
    print(f"  mAP      {result['map']:.1%}")
    print()
    print("  Rank-1 judges the embedding alone: no tracker, no threshold, no")
    print("  binding. IDF1 judges the whole system built on top of it.")


if __name__ == "__main__":
    main()
