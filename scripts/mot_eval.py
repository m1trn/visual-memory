"""Score the system against hand-labelled identities instead of against itself.

Two hypotheses are scored over the same run, because they answer different
questions and only the second is what a viewer sees:

    tracker   the raw track id, which resets every time a track dies
    re-id     the identity number, which is meant to survive that death

The gap between them is the value re-identification actually adds. If binding a
returning object to its earlier record works, ``re-id`` shows fewer identity
switches and a higher IDF1 than ``tracker``; if it binds the wrong record, IDF1
falls and the difference says by how much. No part of this uses the appearance
model to decide who is who, which is what makes it worth running.

Metrics come from ``motmetrics``, the MOT benchmark's own implementation:
IDF1 is a global bipartite matching over a whole sequence and is easy to get
subtly wrong by hand.

Usage: python scripts/mot_eval.py --data data/mot/MOT17/train [--max-frames N]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reid_demo import _fit_verifier, _DEFAULT_CACHE  # noqa: E402
from vision_memory.appearance import build_describer, describe_detections  # noqa: E402
from vision_memory.config import (  # noqa: E402
    load_appearance_config, load_detector_config, load_memory_config,
    load_reid_config, load_tracker_config, load_video_config,
)
from vision_memory.detector import YoloOnnxDetector  # noqa: E402
from vision_memory.memory import VisualMemory  # noqa: E402
from vision_memory.motchallenge import (  # noqa: E402
    Sequence, find_sequences, load_sequence, metrics_module)
from vision_memory.reidentifier import IdentityBinder, ReIdentifier  # noqa: E402
from vision_memory.tracker import ByteTracker  # noqa: E402

# Boxes labelled less visible than this are excluded from scoring: a sliver of
# someone behind a crowd measures the detector's willingness to guess, not the
# system's ability to hold an identity.
_MIN_VISIBILITY = 0.25
# Overlap at which a hypothesis box is considered to be on a labelled person.
# 0.5 is the MOTChallenge convention.
_MATCH_IOU = 0.5
_MIN_EVIDENCE = 2  # observations before a track is worth identifying


def _run(sequence: Sequence, max_frames: int | None) -> tuple[dict, dict, int]:
    """Track and identify through one sequence.

    Returns ``(tracker_ids, reid_ids, identities)``, the first two mapping a
    frame number to ``{hypothesis_id: box}``.
    """
    video_cfg, tracker_cfg = load_video_config(), load_tracker_config()
    reid_cfg = load_reid_config()
    detector = YoloOnnxDetector(load_detector_config())
    describer = build_describer(load_appearance_config())
    verifier, threshold = _fit_verifier(_DEFAULT_CACHE)

    scratch = Path("data/scratch") / f"mot_{sequence.name}"
    scratch.mkdir(parents=True, exist_ok=True)
    mem_cfg = replace(load_memory_config(), db_path=str(scratch / "m.db"),
                      index_path=str(scratch / "m.faiss"))
    for stale in (scratch / "m.db", scratch / "m.faiss"):
        stale.unlink(missing_ok=True)

    tracker = ByteTracker(tracker_cfg)
    by_tracker: dict[int, dict[int, np.ndarray]] = {}
    by_reid: dict[int, dict[int, np.ndarray]] = {}
    fresh = video_cfg.detect_every_n_frames

    with VisualMemory(mem_cfg, describer.dim) as memory:
        reid = ReIdentifier(memory, verifier, threshold=threshold)
        binder = IdentityBinder(reid, sequence.fps, fresh, reid_cfg.reconsider_every, _MIN_EVIDENCE,
                                swap_margin=reid_cfg.swap_margin,
                                min_new_identity_confidence=reid_cfg.min_new_identity_confidence,
                                convincing_confidence=tracker_cfg.high_conf)
        for number, frame in sequence.frames():
            if max_frames is not None and number > max_frames:
                break
            if (number - 1) % video_cfg.detect_every_n_frames == 0:
                detections = detector.detect(frame)
                embeddings = describe_detections(describer, frame, detections)
            else:
                detections, embeddings = None, None
            active = tracker.update(detections, embeddings)
            binder.step(active, number)
            for lost in tracker.pop_lost():
                binder.forget(lost.id)

            drawn = [t for t in active if t.time_since_update <= fresh and t.label == "person"]
            by_tracker[number] = {t.id: t.box for t in drawn}
            # Score what a viewer sees: a numbered box under its identity, an
            # unnumbered one under its track (negative, so the two namespaces
            # cannot collide). A track re-id declines to name is still on screen.
            by_reid[number] = {
                (binder.identity_of(t.id) if binder.identity_of(t.id) is not None else -t.id): t.box
                for t in drawn
            }
        identities = len(memory)
    return by_tracker, by_reid, identities


def _score(sequence: Sequence, hypotheses: dict[int, dict[int, np.ndarray]],
           max_frames: int | None):
    """Accumulate one hypothesis against the sequence's labels."""
    mm = metrics_module()

    acc = mm.MOTAccumulator(auto_id=False)
    for number in sorted(hypotheses):
        if max_frames is not None and number > max_frames:
            break
        truth = sequence.visible(number, _MIN_VISIBILITY)
        hyp = hypotheses[number]
        hyp_ids = list(hyp)
        hyp_boxes = np.array([hyp[i] for i in hyp_ids], dtype=np.float64).reshape(-1, 4)
        # motmetrics measures distance in xywh; 1 - IoU, with anything below the
        # match threshold left as NaN so it cannot be paired at all.
        dist = mm.distances.iou_matrix(
            _to_xywh(truth.boxes), _to_xywh(hyp_boxes), max_iou=1 - _MATCH_IOU
        )
        acc.update(list(truth.ids), hyp_ids, dist, frameid=number)
    return acc


def _to_xywh(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if len(boxes) == 0:
        return boxes
    return np.stack(
        [boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]],
        axis=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/mot"),
                        help="a sequence directory, or any directory containing some")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--sequences", type=int, default=None,
                        help="score only the first N sequences found")
    args = parser.parse_args()

    mm = metrics_module()

    found = find_sequences(args.data)
    if not found:
        raise SystemExit(
            f"no MOT sequences under {args.data}. Each one is a directory holding "
            "seqinfo.ini, img1/ and gt/gt.txt."
        )
    if args.sequences is not None:
        found = found[: args.sequences]

    accs, names = [], []
    for path in found:
        sequence = load_sequence(path)
        if not sequence.truth:
            print(f"{sequence.name}: no gt/gt.txt, skipping")
            continue
        print(f"{sequence.name}: {sequence.length} frames at {sequence.fps:g} fps", flush=True)
        by_tracker, by_reid, identities = _run(sequence, args.max_frames)
        for label, hyp in (("tracker", by_tracker), ("re-id", by_reid)):
            accs.append(_score(sequence, hyp, args.max_frames))
            names.append(f"{sequence.name} {label}")
        print(f"  identities stored: {identities}", flush=True)

    if not accs:
        raise SystemExit("nothing scored")
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs, names=names, generate_overall=False,
        metrics=["idf1", "mota", "motp", "num_switches", "mostly_tracked",
                 "mostly_lost", "num_fragmentations", "num_false_positives", "num_misses"],
    )
    print()
    print(mm.io.render_summary(
        summary, formatters=mh.formatters,
        namemap={"idf1": "IDF1", "mota": "MOTA", "motp": "MOTP",
                 "num_switches": "IDsw", "mostly_tracked": "MT", "mostly_lost": "ML",
                 "num_fragmentations": "Frag", "num_false_positives": "FP",
                 "num_misses": "Miss"},
    ))
    print("\nIDF1 is the number that matters here: how much of each labelled person's")
    print("life was spent under one correct number. IDsw counts how often a number")
    print("jumped to a different person. Compare the two rows per sequence.")


if __name__ == "__main__":
    main()
