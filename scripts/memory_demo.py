"""Phase 6 proof: track a video, store lost tracks as identities, reopen and retrieve.

Usage: python scripts/memory_demo.py [--video PATH] [--max-frames N] [--db PATH] [--index PATH]
(--db/--index default to configs/default.yaml; override them to keep a demo run
out of a real database)
"""

from __future__ import annotations

import argparse
import sqlite3
from contextlib import closing
import sys
import time
import urllib.request
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vision_memory.config import (  # noqa: E402
    load_detector_config, load_encoder_config, load_memory_config,
    load_tracker_config, load_video_config,
)
from vision_memory.detector import YoloOnnxDetector  # noqa: E402
from vision_memory.appearance import AppearanceDescriber  # noqa: E402
from vision_memory.encoder import Encoder  # noqa: E402
from vision_memory.memory import VisualMemory  # noqa: E402
from vision_memory.tracker import ByteTracker  # noqa: E402

_DEFAULT_VIDEO = Path("data/samples/vtest.avi")
_DEFAULT_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi"


def _exemplar_counts(db_path: str) -> dict[int, int]:
    """Read-only peek at the exemplar table so the demo can report index fan-out."""
    with closing(sqlite3.connect(db_path)) as con:
        rows = con.execute("SELECT identity_id, COUNT(*) FROM exemplars GROUP BY identity_id").fetchall()
    return {int(i): int(n) for i, n in rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=_DEFAULT_VIDEO)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--db", type=Path, default=Path("data/demo_memory.db"),
                    help="defaults to a scratch store so a demo never writes the configured one")
    ap.add_argument("--index", type=Path, default=Path("data/demo_memory.faiss"))
    args = ap.parse_args()
    if not args.video.exists():
        args.video.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_DEFAULT_URL, args.video)

    mem_cfg = load_memory_config()
    if args.db is not None:
        mem_cfg = replace(mem_cfg, db_path=str(args.db))
    if args.index is not None:
        mem_cfg = replace(mem_cfg, index_path=str(args.index))
    video_cfg = load_video_config()
    tracker_cfg = load_tracker_config()

    detector = YoloOnnxDetector(load_detector_config())
    describer = AppearanceDescriber(Encoder(load_encoder_config()), load_appearance_config())
    tracker = ByteTracker(tracker_cfg)

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    exemplars: dict[int, list[np.ndarray]] = {}
    first_frame: dict[int, int] = {}
    embed_calls = 0
    embed_time = 0.0
    detect_cycle = 0
    frame_idx = 0
    t_start = time.perf_counter()

    with VisualMemory(mem_cfg, describer.dim) as memory:
        identities_before = len(memory)
        while frame_idx < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            detections = None
            embeddings: dict[int, np.ndarray] = {}
            if frame_idx % video_cfg.detect_every_n_frames == 0:
                detections = detector.detect(frame)
                # Embedding is the expensive stage, so it runs on a slower clock
                # than detection: every Nth detector cycle, batched over the frame.
                if tracker_cfg.embed_every_n and detect_cycle % tracker_cfg.embed_every_n == 0:
                    t0 = time.perf_counter()
                    embeddings = describer.describe(frame, [d.box for d in detections])
                    embed_time += time.perf_counter() - t0
                    embed_calls += len(embeddings)
                detect_cycle += 1

            active = tracker.update(detections, embeddings or None)

            for trk in active:
                first_frame.setdefault(trk.id, frame_idx)

            for lost in tracker.pop_lost():
                # Track.exemplars holds raw observations, already capped and kept
                # spread out by the tracker. The running mean would hand memory a
                # set of near-duplicate smoothed vectors instead.
                if not lost.exemplars:
                    first_frame.pop(lost.id, None)
                    continue
                # A track dies max_age frames after it was last actually seen.
                seen_until = frame_idx - lost.time_since_update
                memory.remember(
                    label=lost.label,
                    embeddings=lost.exemplars,
                    first_seen=first_frame.pop(lost.id, frame_idx) / fps,
                    last_seen=seen_until / fps,
                    appearances=lost.hits,
                )
            frame_idx += 1

        wall = time.perf_counter() - t_start
        stored = len(memory) - identities_before
        memory.save()
    cap.release()

    # Reopening from disk is the whole point of Phase 6: identities must survive
    # the process, not just live in RAM.
    with VisualMemory(mem_cfg, describer.dim) as memory:
        identities = memory.all_identities()
        print(f"reopened {mem_cfg.db_path} -> {len(memory)} identities")
        print(f"{'id':>4}  {'label':<12} {'appear':>6} {'seconds':>8} {'exemplars':>9}")
        counts = _exemplar_counts(mem_cfg.db_path)
        total_exemplars = sum(counts.values())
        for ident in identities:
            print(f"{ident.id:>4}  {ident.label:<12} {ident.appearances:>6} "
                  f"{ident.last_seen - ident.first_seen:>8.2f} {counts.get(ident.id, 0):>9}")
        if identities:
            probe = identities[0]
            print(f"\nsearch with identity #{probe.id} prototype:")
            for rank, (ident_id, score) in enumerate(memory.search(probe.prototype, k=3), 1):
                match = memory.get(ident_id)
                label = match.label if match else "?"
                print(f"  {rank}. identity #{ident_id} ({label}) cosine={score:.3f}")

    ms = embed_time / embed_calls * 1000 if embed_calls else 0.0
    print(f"\n{args.video.name}: {frame_idx} frames in {wall:.1f}s")
    print(f"  identities stored: {stored}   total exemplars: {total_exemplars}")
    print(f"  embed calls: {embed_calls} ({ms:.0f} ms/embed)")


if __name__ == "__main__":
    main()
