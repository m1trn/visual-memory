"""Phase 7 proof: lost tracks are resolved against persistent memory and re-bound.

When a track dies, ReIdentifier searches memory and either binds it into an
existing identity or creates a new one; frames are written with a short lag so a
track's boxes can carry the identity it turned out to be. The reid_bench pair
cache, when present, supplies the pairs the threshold is learned from.

Usage: python scripts/reid_demo.py [--video PATH] [--max-frames N] [--db PATH] [--index PATH]
(--db/--index default to scratch paths so a demo never writes the configured store)
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reid_bench import _DEFAULT_CACHE, crop_rgb  # noqa: E402
from track_demo import _color  # noqa: E402
from vision_memory.config import (  # noqa: E402
    load_detector_config, load_encoder_config, load_memory_config, load_reid_config,
    load_tracker_config, load_video_config,
)
from vision_memory.detector import YoloOnnxDetector  # noqa: E402
from vision_memory.encoder import Encoder  # noqa: E402
from vision_memory.memory import VisualMemory  # noqa: E402
from vision_memory.reid import Verifier, balance, build_verifier, mine_pairs, split_by_group  # noqa: E402
from vision_memory.reidentifier import ReIdentifier, Resolution  # noqa: E402
from vision_memory.tracker import ByteTracker  # noqa: E402

_DEFAULT_VIDEO = Path("data/samples/vtest.avi")
_DEFAULT_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi"
_Boxes = list[tuple[int, np.ndarray, str]]


def _fit_verifier(pairs_path: Path) -> tuple[Verifier, bool]:
    """Build the verifier, learning its threshold from cached pairs when they exist."""
    cfg = load_reid_config()
    verifier = build_verifier(cfg)
    if not pairs_path.exists():
        raise SystemExit(
            f"no mined pairs at {pairs_path}; run scripts/reid_bench.py first "
            "so the decision threshold is learned from data rather than guessed"
        )
    rng = np.random.default_rng(cfg.seed)
    observations, seen_on = pickle.loads(pairs_path.read_bytes())
    a, b, y, owners = mine_pairs(observations, cfg, rng, frames=seen_on)
    train_mask, _ = split_by_group(y, owners, cfg.test_fraction, rng)
    train_mask = balance(y, train_mask, rng)
    if not train_mask.any() or len(np.unique(y[train_mask])) < 2:
        raise SystemExit("mined pairs do not contain both classes; try a longer clip")
    verifier.fit(a[train_mask], b[train_mask], y[train_mask])
    return verifier, True


def _flush(writer: cv2.VideoWriter, pending: list[tuple[np.ndarray, _Boxes]],
           bound: dict[int, Resolution], keep: int) -> None:
    """Annotate and write buffered frames until only ``keep`` remain pending."""
    while len(pending) > keep:
        frame, boxes = pending.pop(0)
        for track_id, box, label in boxes:
            res = bound.get(track_id)
            color = _color(res.identity_id if res else track_id)
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"#{track_id} {label}" if res is None else (
                f"#{res.identity_id} {label} "
                + ("new" if res.is_new else f"seen before ({res.score:.2f})"))
            cv2.putText(frame, text, (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        writer.write(frame)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=_DEFAULT_VIDEO)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--db", type=Path, default=Path("data/reid_demo.db"))
    ap.add_argument("--index", type=Path, default=Path("data/reid_demo.faiss"))
    args = ap.parse_args()
    if not args.video.exists():
        args.video.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_DEFAULT_URL, args.video)

    video_cfg, tracker_cfg = load_video_config(), load_tracker_config()
    mem_cfg = replace(load_memory_config(), db_path=str(args.db), index_path=str(args.index))
    detector, encoder = YoloOnnxDetector(load_detector_config()), Encoder(load_encoder_config())
    tracker = ByteTracker(tracker_cfg)
    verifier, learned = _fit_verifier(_DEFAULT_CACHE)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out_path = Path("data/tracking") / f"{args.video.stem}_reid.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    # A track is only resolved once it dies, max_age frames after its last
    # sighting, so frames are held back that long before being written.
    lag, fresh = tracker_cfg.max_age + 1, video_cfg.detect_every_n_frames
    pending: list[tuple[np.ndarray, _Boxes]] = []
    bound: dict[int, Resolution] = {}
    first_frame: dict[int, int] = {}
    lost_count = rebound = created = frame_idx = 0
    with VisualMemory(mem_cfg, encoder.dim) as memory:
        reid = ReIdentifier(memory, verifier)
        while frame_idx < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            detections = detector.detect(frame) if frame_idx % video_cfg.detect_every_n_frames == 0 else None
            active = tracker.update(detections)
            crops = [(t, crop_rgb(frame, t.box, video_cfg.min_crop_px)) for t in tracker.due_for_embedding()]
            usable = [(t, c) for t, c in crops if c is not None]
            if usable:
                for (track, _), vec in zip(usable, encoder.encode_batch([c for _, c in usable])):
                    tracker.add_embedding(track, vec)
            first_frame.update({t.id: frame_idx for t in active if t.id not in first_frame})
            # A track coasting on prediction alone gets no box drawn.
            pending.append((frame, [(t.id, t.box.copy(), t.label) for t in active if t.time_since_update <= fresh]))

            for lost in tracker.pop_lost():
                lost_count += 1
                if not lost.exemplars:
                    first_frame.pop(lost.id, None)
                    continue
                res = reid.resolve(lost.label, lost.exemplars,
                                   first_frame.pop(lost.id, frame_idx) / fps,
                                   (frame_idx - lost.time_since_update) / fps, lost.hits)
                bound[lost.id] = res
                created += res.is_new
                rebound += not res.is_new

            _flush(writer, pending, bound, lag)
            frame_idx += 1

        _flush(writer, pending, bound, 0)
        identity_count = len(memory)
        memory.save()
    cap.release()
    writer.release()

    print(f"{args.video.name}: {frame_idx} frames, threshold {verifier.threshold:.3f} "
          f"({'learned from mined pairs' if learned else 'config default'})")
    print(f"  tracks lost: {lost_count}   bound to existing: {rebound}   created new: {created}")
    print(f"  identities in {mem_cfg.db_path}: {identity_count}\nannotated -> {out_path}")


if __name__ == "__main__":
    main()
