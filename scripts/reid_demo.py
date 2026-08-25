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
import os
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
from vision_memory.reid import (Verifier, balance, build_verifier,
                                calibrate_identity_threshold, mine_pairs, split_by_group)  # noqa: E402
from vision_memory.reidentifier import ReIdentifier, Resolution  # noqa: E402
from vision_memory.tracker import ByteTracker  # noqa: E402

_DEFAULT_VIDEO = Path("data/samples/vtest.avi")
_DEFAULT_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi"
_Boxes = list[tuple[int, np.ndarray, str]]
_MIN_EVIDENCE = 2  # observations before a track is worth identifying


def _fit_verifier(pairs_path: Path) -> tuple[Verifier, float]:
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
    # The verifier's own boundary is fitted to pair scores; the binding decision
    # compares an aggregate, so it needs a boundary fitted to that instead.
    threshold, n_pos, n_neg = calibrate_identity_threshold(
        observations, seen_on, cfg.observation_quantile,
        load_memory_config().exemplars_per_identity, cfg.max_false_merge_rate,
    )
    print(f"calibrated on {n_pos} same-object and {n_neg} provably-different track/identity examples")
    return verifier, threshold


def _draw(frame: np.ndarray, boxes: _Boxes, bound: dict[int, Resolution]) -> None:
    """Label every box with its memory identity.

    A track and an identity are different numbering systems, so a box must never
    show one and then the other — the same person appearing first as track #12
    and later as identity #12 (a different person entirely) reads exactly like
    the tracker swapping their ids. Every box here carries the identity, which
    is assigned as soon as the track is confirmed and never changes afterwards.
    """
    for track_id, box, label in boxes:
        res = bound.get(track_id)
        if res is None:
            continue  # not yet identified; drawing a provisional number would lie
        color = _color(res.identity_id)
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"#{res.identity_id} {label} " + ("new" if res.is_new else f"seen before ({res.score:.2f})")
        cv2.putText(frame, text, (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


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
    verifier, threshold = _fit_verifier(_DEFAULT_CACHE)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out_path = Path("data/tracking") / f"{args.video.stem}_reid.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write under a temporary name and rename only once the container has been
    # finalized. An interrupted run otherwise leaves an mp4 with no moov atom,
    # which no player will open, in place of the last good one.
    part_path = out_path.with_suffix(".part.mp4")
    writer = cv2.VideoWriter(str(part_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    fresh = video_cfg.detect_every_n_frames
    bound: dict[int, Resolution] = {}
    first_frame: dict[int, int] = {}
    lost_count = rebound = created = frame_idx = 0
    with VisualMemory(mem_cfg, encoder.dim) as memory:
        reid = ReIdentifier(memory, verifier, threshold=threshold)
        while frame_idx < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            detections = detector.detect(frame) if frame_idx % video_cfg.detect_every_n_frames == 0 else None
            # Embed the detections BEFORE association, so appearance can help
            # decide who is who rather than only being recorded afterwards.
            # Without this the tracker arbitrates a crossing on box position
            # alone, which is how one person ends up with another's id.
            embeddings = None
            if detections:
                crops, idx = [], []
                for j, det in enumerate(detections):
                    crop = crop_rgb(frame, det.box, video_cfg.min_crop_px, video_cfg.crop_upper_fraction)
                    if crop is not None:
                        idx.append(j)
                        crops.append(crop)
                if crops:
                    embeddings = dict(zip(idx, encoder.encode_batch(crops)))
            active = tracker.update(detections, embeddings)
            first_frame.update({t.id: frame_idx for t in active if t.id not in first_frame})

            # Identify a track as soon as it has enough appearance evidence,
            # while it is still being watched, rather than waiting for it to die.
            # This is what makes the number on screen stable from the moment the
            # object appears, and it is also how a live system has to work: you
            # cannot tell a viewer who somebody is only once they have left.
            for track in active:
                if track.id in bound or len(track.exemplars) < _MIN_EVIDENCE:
                    continue
                res = reid.resolve(track.label, track.exemplars,
                                   first_frame.get(track.id, frame_idx) / fps,
                                   frame_idx / fps, track.hits, box=track.box)
                bound[track.id] = res
                created += res.is_new
                rebound += not res.is_new

            # Keep every identified object's whereabouts current, so a track that
            # dies here can lend its continuity to one appearing here next.
            for track in active:
                res = bound.get(track.id)
                if res is not None and track.time_since_update == 0:
                    reid.note_seen(res.identity_id, frame_idx / fps, track.box)

            # A track coasting on prediction alone gets no box drawn.
            _draw(frame, [(t.id, t.box.copy(), t.label) for t in active if t.time_since_update <= fresh], bound)
            writer.write(frame)

            for lost in tracker.pop_lost():
                lost_count += 1
                res = bound.get(lost.id)
                first = first_frame.pop(lost.id, frame_idx) / fps
                last = (frame_idx - lost.time_since_update) / fps
                if res is not None and lost.exemplars:
                    # Fold everything the track ended up seeing into the identity
                    # it was already given, so memory keeps the better record
                    # without the number on screen ever changing.
                    memory.remember(lost.label, lost.exemplars, first, last,
                                    lost.hits, identity_id=res.identity_id)

            frame_idx += 1

        identity_count = len(memory)
        memory.save()
    cap.release()
    writer.release()
    os.replace(part_path, out_path)

    print(f"{args.video.name}: {frame_idx} frames, threshold {threshold:.3f} (calibrated on track/identity scores)")
    print(f"  tracks identified: {rebound + created}   recognized: {rebound}   new: {created}"
          f"   (of {lost_count} that later ended)")
    print(f"  identities in {mem_cfg.db_path}: {identity_count}\nannotated -> {out_path}")


if __name__ == "__main__":
    main()
