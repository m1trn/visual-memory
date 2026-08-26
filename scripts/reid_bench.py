"""Phase 7 evidence: does a learned verification threshold beat the hand-set one?

Tracks a video, embeds each track's crops, mines same-track/different-track
pairs, splits them BY TRACK (never randomly) and reports held-out metrics for
each verifier plus the configured cosine threshold as a baseline.

Usage: python scripts/reid_bench.py [--video PATH] [--max-frames N] [--refresh]
"""

from __future__ import annotations

import argparse
import pickle
import sys
import urllib.request
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vision_memory.config import (  # noqa: E402
    load_appearance_config,
    load_detector_config, load_encoder_config, load_memory_config,
    load_reid_config, load_tracker_config, load_video_config,
)
from vision_memory.detector import YoloOnnxDetector  # noqa: E402
from vision_memory.appearance import AppearanceDescriber, build_embedder  # noqa: E402
from vision_memory.encoder import Encoder  # noqa: E402
from vision_memory.metrics import auroc  # noqa: E402
from vision_memory.reid import balance, build_verifier, mine_pairs, split_by_group  # noqa: E402
from vision_memory.tracker import ByteTracker  # noqa: E402

_DEFAULT_VIDEO = Path("data/samples/vtest.avi")
_DEFAULT_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi"
_DEFAULT_CACHE = Path("data/reid_pairs.pkl")


def collect_track_embeddings(
    video: Path, max_frames: int
) -> tuple[dict[int, list[np.ndarray]], dict[int, list[int]]]:
    """Track ``video``, returning per-track embeddings and the frame each was seen on.

    The frame indices are what let mining prove a negative pair: two tracks
    alive in the same frame cannot be the same object. Without them, two
    fragments of one person get labelled "different" and poison the training set.
    """
    video_cfg = load_video_config()
    detector = YoloOnnxDetector(load_detector_config())
    describer = AppearanceDescriber(*_appearance())
    tracker = ByteTracker(load_tracker_config())

    observations: dict[int, list[np.ndarray]] = {}
    seen_on: dict[int, list[int]] = {}
    cap = cv2.VideoCapture(str(video))
    frame_idx = 0
    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        detections = detector.detect(frame) if frame_idx % video_cfg.detect_every_n_frames == 0 else None
        # Describe before associating, and with the same descriptor the rest of
        # the system uses, so what is measured here is what actually ships.
        described = describer.describe(frame, [d.box for d in detections]) if detections else None
        active = tracker.update(detections, described)
        # Record a vector against the track that ended up owning that detection.
        for track in active:
            if track.time_since_update == 0 and track.exemplars:
                observations.setdefault(track.id, []).append(
                    np.asarray(track.exemplars[-1], dtype=np.float32))
                seen_on.setdefault(track.id, []).append(frame_idx)
        frame_idx += 1
    cap.release()
    return observations, seen_on


def _quantiles(values: np.ndarray) -> str:
    """min / 25th / median / 75th / max of ``values`` as a fixed-width row."""
    q = np.percentile(values, [0, 25, 50, 75, 100])
    return "".join(f"{v:>9.3f}" for v in q)


def _rates(y_true: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    """Accuracy, true-positive rate and true-negative rate of ``pred`` against ``y_true``.

    TPR and TNR are reported rather than precision because they are read
    independently of the class balance of whatever split they land on.
    """
    truth = y_true.astype(bool)
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    return (float((pred == truth).mean()) if len(truth) else 0.0,
            tp / (tp + fn) if tp + fn else 0.0,   # TPR: true matches accepted
            tn / (tn + fp) if tn + fp else 0.0)   # TNR: different objects rejected


def _appearance() -> tuple:
    """The configured appearance model, ready to describe boxes."""
    cfg = load_appearance_config()
    embedder, upper = build_embedder(cfg)
    return embedder, cfg, upper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=_DEFAULT_VIDEO)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--cache", type=Path, default=_DEFAULT_CACHE)
    ap.add_argument("--refresh", action="store_true", help="re-run the video instead of using the cache")
    args = ap.parse_args()
    if not args.video.exists():
        args.video.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_DEFAULT_URL, args.video)

    if args.cache.exists() and not args.refresh:
        observations, seen_on = pickle.loads(args.cache.read_bytes())
        print(f"loaded {args.cache} ({len(observations)} tracks)")
    else:
        observations, seen_on = collect_track_embeddings(args.video, args.max_frames)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_bytes(pickle.dumps((observations, seen_on)))
        print(f"tracked {args.video.name} -> {len(observations)} tracks, cached to {args.cache}")

    cfg = load_reid_config()
    rng = np.random.default_rng(cfg.seed)
    a, b, y, owners = mine_pairs(observations, cfg, rng, frames=seen_on)
    if len(y) == 0:
        raise SystemExit("no pairs mined; try a longer --max-frames or a lower reid.min_pair_gap")
    train_mask, test_mask = split_by_group(y, owners, cfg.test_fraction, rng)
    # Mining balances the population, but a split does not preserve that, and
    # raw accuracy on a skewed side measures the prior rather than the verifier.
    train_mask, test_mask = balance(y, train_mask, rng), balance(y, test_mask, rng)
    if not train_mask.any() or not test_mask.any():
        raise SystemExit('not enough tracks to hold any out; try a longer clip')
    held_out = sorted(set(np.unique(owners[test_mask]).tolist()))
    print(f"pairs: {len(y)} mined from {len(np.unique(owners))} tracks; "
          f"train {int(train_mask.sum())} / test {int(test_mask.sum())} (balanced)")
    print(f"held-out tracks (unseen in training): {held_out}")

    cosine = np.sum(a * b, axis=1)
    print(f"\ncosine distribution{'min':>7}{'p25':>9}{'median':>9}{'p75':>9}{'max':>9}")
    print(f"  same track      {_quantiles(cosine[y == 1])}")
    print(f"  different track {_quantiles(cosine[y == 0])}")

    y_test, a_test, b_test = y[test_mask], a[test_mask], b[test_mask]
    print(f"\n{'verifier':<20}{'threshold':>10}{'AUROC':>8}{'acc':>8}{'TPR':>8}{'TNR':>8}")
    for name in ("cosine", "logistic"):
        verifier = build_verifier(replace(cfg, verifier=name))
        verifier.fit(a[train_mask], b[train_mask], y[train_mask])
        scores = verifier.score(a_test, b_test)
        accuracy, tpr, tnr = _rates(y_test, verifier.predict(a_test, b_test))
        print(f"{name + ' (learned)':<20}{verifier.threshold:>10.3f}{auroc(scores, y_test):>8.3f}"
              f"{accuracy:>8.3f}{tpr:>8.3f}{tnr:>8.3f}")

    # The hand-set baseline: same cosine score, config threshold instead of a
    # learned one. AUROC is identical by construction; only the boundary moves.
    hand_set = load_memory_config().reid_threshold
    test_cos = cosine[test_mask]
    accuracy, tpr, tnr = _rates(y_test, test_cos >= hand_set)
    print(f"{'cosine @ 0.80 (fixed)':<20}{hand_set:>10.3f}{auroc(test_cos, y_test):>8.3f}"
          f"{accuracy:>8.3f}{tpr:>8.3f}{tnr:>8.3f}")


if __name__ == "__main__":
    main()
