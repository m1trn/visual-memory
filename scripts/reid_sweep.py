"""Choose the re-id operating point by what it does to identity, not by argument.

`max_false_merge_rate` was set to 2% on the reasoning that a wrong merge
corrupts a record permanently while a missed return only costs a spare
identity. That reasoning is sound and was never measured. With labelled data it
can be: each candidate threshold is run end to end and scored against the
annotator's identities, so the choice is made on IDF1 and identity switches
rather than on a rate nobody has connected to either.

Detection and embedding are cached once, because they dominate the runtime and
do not depend on the threshold. Everything downstream — tracking, memory,
binding — is re-run per threshold.

Usage: python scripts/reid_sweep.py [--data data/mot] [--max-frames N]
"""

from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mot_eval import _MIN_EVIDENCE, _score  # noqa: E402
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

CACHE = Path("data/mot/detection_cache.pkl")


class _Fixed:
    """A verifier whose boundary is set by the sweep rather than fitted."""

    def __init__(self, threshold: float) -> None:
        self._t = threshold

    def fit(self, *a: object) -> None:
        pass

    def score(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (np.asarray(a) * np.asarray(b)).sum(1)

    @property
    def threshold(self) -> float:
        return self._t

    def predict(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self.score(a, b) >= self._t


def build_cache(sequence: Sequence, max_frames: int | None) -> list:
    """Detections and embeddings per frame; the part that does not vary."""
    video_cfg = load_video_config()
    detector = YoloOnnxDetector(load_detector_config())
    describer = build_describer(load_appearance_config())
    frames = []
    for number, frame in sequence.frames():
        if max_frames is not None and number > max_frames:
            break
        if (number - 1) % video_cfg.detect_every_n_frames == 0:
            detections = detector.detect(frame)
            embeddings = describe_detections(describer, frame, detections, load_tracker_config().min_exemplar_confidence)
        else:
            detections, embeddings = None, None
        frames.append((number, detections, embeddings))
        if number % 100 == 0:
            print(f"  cached {number}", flush=True)
    return frames


def run(frames: list, sequence: Sequence, threshold: float | None, dim: int,
        labels: "frozenset[str]" = frozenset({"person"})):
    """Track, and optionally identify, returning per-frame hypothesis boxes."""
    video_cfg, tracker_cfg = load_video_config(), load_tracker_config()
    fresh = video_cfg.detect_every_n_frames
    tracker = ByteTracker(tracker_cfg)
    by_tracker: dict[int, dict[int, np.ndarray]] = {}
    by_reid: dict[int, dict[int, np.ndarray]] = {}

    scratch = Path("data/scratch/sweep")
    scratch.mkdir(parents=True, exist_ok=True)
    for stale in (scratch / "m.db", scratch / "m.faiss"):
        stale.unlink(missing_ok=True)
    mem_cfg = replace(load_memory_config(), db_path=str(scratch / "m.db"),
                      index_path=str(scratch / "m.faiss"))

    with VisualMemory(mem_cfg, dim) as memory:
        reid = ReIdentifier(memory, _Fixed(threshold or 1.0), threshold=threshold or 1.0)
        rc = load_reid_config()
        binder = IdentityBinder(reid, sequence.fps, fresh, rc.reconsider_every, _MIN_EVIDENCE,
                                swap_margin=rc.swap_margin,
                                min_new_identity_confidence=rc.min_new_identity_confidence,
                                convincing_confidence=tracker_cfg.high_conf,
                                recent_views=tracker_cfg.veto_views,
                                still_object_motion=rc.still_object_motion)
        for number, detections, embeddings in frames:
            active = tracker.update(detections, embeddings)
            if threshold is not None:
                binder.step(active, number)
            for lost in tracker.pop_lost():
                binder.forget(lost.id)

            drawn = [t for t in active if t.time_since_update <= fresh and t.label in labels]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/mot"))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--classes", type=int, nargs="+", default=None,
                        help="annotated class ids to score (default: MOT17 pedestrians). "
                             "VisDrone vehicles are 4 car, 5 van, 6 truck, 9 bus, 10 motor.")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="detector labels to track (default: person)")
    args = parser.parse_args()

    mm = metrics_module()
    sequences = [load_sequence(p, tuple(args.classes) if args.classes else None)
                 for p in find_sequences(args.data)]
    sequences = [s for s in sequences if s.truth]
    if not sequences:
        raise SystemExit(f"no labelled sequences under {args.data}")
    sequence = sequences[0]

    cache = args.cache or CACHE
    if cache.exists() and not args.rebuild:
        frames = pickle.loads(cache.read_bytes())
        print(f"{sequence.name}: {len(frames)} frames from cache")
    else:
        print(f"{sequence.name}: caching detections and embeddings", flush=True)
        frames = build_cache(sequence, args.max_frames)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps(frames))

    dim = build_describer(load_appearance_config()).dim
    fitted = load_reid_config()

    accs, names = [], []
    identities = {}
    labels = frozenset(args.labels or ('person',))
    by_tracker, _, _ = run(frames, sequence, None, dim, labels)
    accs.append(_score(sequence, by_tracker, args.max_frames))
    names.append("tracker only")

    for threshold in (0.55, 0.593, 0.65):
        _, by_reid, count = run(frames, sequence, threshold, dim, labels)
        accs.append(_score(sequence, by_reid, args.max_frames))
        names.append(f"re-id @ {threshold:.3f}")
        identities[threshold] = count
        print(f"  {threshold:.3f}: {count} identities", flush=True)

    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs, names=names, generate_overall=False,
        metrics=["idf1", "mota", "num_switches", "mostly_tracked", "num_fragmentations"],
    )
    print()
    print(mm.io.render_summary(
        summary, formatters=mh.formatters,
        namemap={"idf1": "IDF1", "mota": "MOTA", "num_switches": "IDsw",
                 "mostly_tracked": "MT", "num_fragmentations": "Frag"},
    ))
    print("\nThe tracker row is the floor: it is what the numbers look like with no")
    print("re-identification at all. A threshold only earns its place by beating it.")
    print(f"(currently configured boundary: {0.704:.3f}, "
          f"max_false_merge_rate {fitted.max_false_merge_rate:.0%})")


if __name__ == "__main__":
    main()
