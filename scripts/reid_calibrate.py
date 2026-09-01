"""Learn the re-id boundary from real returns, using hand-labelled identities.

`calibrate_identity_threshold` builds its positives by splitting one track into
two adjacent halves, so the query's first observation is a fraction of a second
after the stored one's last. That is not the situation re-identification meets.
Measured on this footage the score decays with the gap — median 0.858 at 0-2s
against 0.770 at 4-6s, with the 5th percentile collapsing 0.764 -> 0.467 — so
the boundary is fitted on easier examples than it is asked to judge, and the
resulting threshold sits above most of what a genuine return actually scores.

With a labelled sequence there is no need to approximate. Every observation is
attributed to the person the annotator says it belongs to, so a positive is
literally what re-id exists to catch: the same labelled person, seen again after
our own tracker lost them. A negative is two different labelled people, which
needs no co-aliveness argument because the labels settle it outright.

Usage:
    python scripts/reid_calibrate.py --data data/mot          # build cache and fit
    python scripts/reid_calibrate.py --report                 # re-fit from cache only
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vision_memory.appearance import build_describer, describe_detections  # noqa: E402
from vision_memory.config import (  # noqa: E402
    load_appearance_config, load_detector_config, load_memory_config,
    load_reid_config, load_tracker_config, load_video_config,
)
from vision_memory.detector import YoloOnnxDetector  # noqa: E402
from vision_memory.motchallenge import find_sequences, load_sequence  # noqa: E402
from vision_memory.reid import identity_score, select_diverse  # noqa: E402
from vision_memory.tracker import ByteTracker, iou_matrix  # noqa: E402

CACHE = Path("data/mot/labelled_observations.pkl")
_ATTRIBUTION_IOU = 0.5  # overlap at which an observation belongs to a labelled person
_MIN_VISIBILITY = 0.25


def build_cache(data: Path, max_frames: int | None) -> list[dict]:
    """Run the pipeline over labelled sequences, tagging each observation.

    Each record is one embedding, together with the labelled person it belongs
    to and the tracker track it arrived on. The pair of those two is what makes
    a real return identifiable: the same person on a *different* track is
    exactly a case the tracker lost and re-id must recover.
    """
    video_cfg = load_video_config()
    detector = YoloOnnxDetector(load_detector_config())
    describer = build_describer(load_appearance_config())

    records: list[dict] = []
    for path in find_sequences(data):
        sequence = load_sequence(path)
        if not sequence.truth:
            continue
        print(f"{sequence.name}: {sequence.length} frames", flush=True)
        tracker = ByteTracker(load_tracker_config())
        for number, frame in sequence.frames():
            if max_frames is not None and number > max_frames:
                break
            if (number - 1) % video_cfg.detect_every_n_frames == 0:
                detections = detector.detect(frame)
                embeddings = describe_detections(describer, frame, detections, load_tracker_config().min_exemplar_confidence)
            else:
                detections, embeddings = None, None
            active = tracker.update(detections, embeddings)

            truth = sequence.visible(number, _MIN_VISIBILITY)
            if len(truth.ids) == 0:
                continue
            fresh = [t for t in active if t.time_since_update == 0
                     and t.label == "person" and t.exemplars]
            if not fresh:
                continue
            boxes = np.array([t.box for t in fresh], dtype=np.float64)
            overlap = iou_matrix(boxes, truth.boxes)
            for row, track in enumerate(fresh):
                best = int(overlap[row].argmax())
                if overlap[row, best] < _ATTRIBUTION_IOU:
                    continue  # not confidently anyone the annotator labelled
                records.append({
                    "sequence": sequence.name,
                    "frame": number,
                    "person": int(truth.ids[best]),
                    "track": track.id,
                    "embedding": np.asarray(track.exemplars[-1], dtype=np.float32),
                })
        print(f"  {len(records)} attributed observations so far", flush=True)
    return records


def _by_person(records: list[dict]) -> dict[tuple[str, int], list[dict]]:
    out: dict[tuple[str, int], list[dict]] = {}
    for r in records:
        out.setdefault((r["sequence"], r["person"]), []).append(r)
    return out


def mine(records: list[dict], exemplars: int, quantile: float
         ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Score every real return, and every different-person comparison.

    A positive is one labelled person's observations on a later track, scored
    against a memory built from their observations on an earlier one — the exact
    question `resolve` is asked. A negative is that same query against a
    different labelled person.
    """
    people = _by_person(records)
    stored: dict[tuple[str, int], np.ndarray] = {}
    queries: dict[tuple[str, int], np.ndarray] = {}

    returns = 0
    for key, rows in people.items():
        tracks: dict[int, list[dict]] = {}
        for r in rows:
            tracks.setdefault(r["track"], []).append(r)
        if len(tracks) < 2:
            continue  # never lost, so never re-identified
        order = sorted(tracks, key=lambda t: min(x["frame"] for x in tracks[t]))
        first = np.stack([x["embedding"] for x in tracks[order[0]]])
        later = np.stack([x["embedding"] for x in tracks[order[-1]]])
        if len(first) < 2 or len(later) < 2:
            continue
        stored[key] = first[select_diverse(first, exemplars)]
        queries[key] = later
        returns += 1

    pos, neg = [], []
    for key, query in queries.items():
        pos.append(identity_score(query, stored[key], quantile))
        for other, memory in stored.items():
            if other == key or other[0] != key[0]:
                continue
            neg.append(identity_score(query, memory, quantile))
    info = {"people": len(people), "returns": returns}
    return np.array(pos, np.float32), np.array(neg, np.float32), info


def threshold_at(pos: np.ndarray, neg: np.ndarray, rate: float) -> float:
    """Lowest boundary holding wrong merges at or under ``rate``."""
    if len(neg) == 0:
        return float("inf")
    ordered = np.sort(neg)
    allowed = int(np.floor(rate * len(ordered)))
    if allowed >= len(ordered):
        return float(ordered[0])
    return float(ordered[len(ordered) - 1 - allowed]) + 1e-6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/mot"))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--cache", type=Path, default=None,
                        help="write/read the observation cache, keeping tuning and "
                             "held-out sequences apart")
    parser.add_argument("--report", action="store_true",
                        help="fit from the existing cache without re-running the pipeline")
    args = parser.parse_args()

    cache = args.cache or CACHE
    if args.report and cache.exists():
        records = pickle.loads(cache.read_bytes())
    else:
        records = build_cache(args.data, args.max_frames)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps(records))
    if not records:
        raise SystemExit("no attributed observations; is there labelled data under --data?")

    reid_cfg = load_reid_config()
    exemplars = load_memory_config().exemplars_per_identity
    pos, neg = mine(records, exemplars, reid_cfg.observation_quantile)[:2]
    info = mine(records, exemplars, reid_cfg.observation_quantile)[2]

    print(f"\n{len(records)} observations attributed to {info['people']} labelled people")
    print(f"{info['returns']} of them were lost by the tracker and seen again — the real returns\n")
    if len(pos) == 0:
        raise SystemExit("no real returns found; try a longer sequence")

    print(f"{'':<26}{'median':>9}{'5th pct':>10}{'95th pct':>10}{'n':>7}")
    print(f"{'same person, seen again':<26}{np.median(pos):>9.3f}"
          f"{np.percentile(pos, 5):>10.3f}{'':>10}{len(pos):>7}")
    print(f"{'different people':<26}{np.median(neg):>9.3f}{'':>10}"
          f"{np.percentile(neg, 95):>10.3f}{len(neg):>7}")

    lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    order = np.argsort(-np.r_[pos, neg])
    lab = lab[order]
    tp, fp = np.cumsum(lab), np.cumsum(1 - lab)
    auroc = float(np.trapezoid(tp / max(tp[-1], 1), fp / max(fp[-1], 1)))
    print(f"\nAUROC on real returns: {auroc:.3f}")

    fitted = threshold_at(pos, neg, reid_cfg.max_false_merge_rate)
    print(f"\n{'threshold':>10}{'returns recovered':>20}{'wrong merges':>15}")
    for t in sorted({0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.704, 0.75, round(fitted, 3)}):
        mark = "  <- fitted here" if abs(t - round(fitted, 3)) < 1e-9 else ""
        print(f"{t:>10.3f}{(pos >= t).mean():>19.0%}{(neg >= t).mean():>14.1%}{mark}")
    print(f"\nfitted threshold at {reid_cfg.max_false_merge_rate:.0%} wrong merges: {fitted:.3f}")
    print("currently in use, fitted on same-track splits: 0.704")


if __name__ == "__main__":
    main()
