"""Phase 8: the whole system on a camera, three modes over one engine.

    1  MEMORY   every object wears its identity number; returning objects get theirs back
    2  SEARCH   click an object: the stored identities that look most like it
    3  ANOMALY  each object coloured by how unusual it looks for its kind
    q  quit

Two threads. The pipeline (detect -> embed -> track -> identify) takes a few
hundred milliseconds per frame on this CPU, so it runs on a worker that always
takes the NEWEST camera frame when it is free - the offline pipeline with an
adaptive frame skip, and the code the labelled evaluation scored. The display
draws every camera frame at camera rate, projecting each box forward from the
last processed frame with the tracker's own motion model, so boxes move
smoothly and numbers never change between updates.

Usage: python scripts/live_demo.py [--source 0 | path/to/video] [--db PATH] [--index PATH]
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reid_demo import _DEFAULT_CACHE, _color, _dashed, _fit_verifier  # noqa: E402
from vision_memory.config import (  # noqa: E402
    load_anomaly_config, load_appearance_config, load_detector_config, load_memory_config,
    load_reid_config, load_tracker_config, load_video_config,
)
from vision_memory.engine import TrackView, VisionEngine  # noqa: E402

MODES = {ord("1"): "memory", ord("2"): "search", ord("3"): "anomaly"}


class LatestFrame:
    """A one-slot mailbox: the newest camera frame, and how many have arrived."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self.index = -1
        self.closed = False

    def put(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame
            self.index += 1

    def get(self) -> tuple[np.ndarray | None, int]:
        with self._lock:
            return (None if self._frame is None else self._frame.copy()), self.index


def _capture(source, mailbox: LatestFrame, pace_fps: float | None) -> None:
    """Read frames into the mailbox. A file is paced to its own frame rate."""
    cap = cv2.VideoCapture(source)
    period = 1.0 / pace_fps if pace_fps else 0.0
    while not mailbox.closed:
        t = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        mailbox.put(frame)
        if period:
            time.sleep(max(period - (time.perf_counter() - t), 0.0))
    cap.release()
    mailbox.closed = True


def _worker(engine: VisionEngine, mailbox: LatestFrame, stats: dict) -> None:
    """Process the newest frame whenever free; skip whatever arrived meanwhile."""
    done = -1
    while not mailbox.closed:
        frame, idx = mailbox.get()
        if frame is None or idx == done:
            time.sleep(0.002)
            continue
        t = time.perf_counter()
        try:
            engine.process(frame, idx)
        except Exception:  # a silent daemon-thread death would look like a freeze
            import traceback
            traceback.print_exc()
            stats["error"] = True
            return
        stats["pipeline_ms"] = (time.perf_counter() - t) * 1000
        stats["skipped"] = idx - done - 1 if done >= 0 else 0
        done = idx


def _draw_memory(frame, views: list[TrackView]) -> None:
    for v in views:
        if v.identity_id is None:
            continue
        colour = _color(v.identity_id)
        x1, y1, x2, y2 = (int(c) for c in v.box)
        if v.hidden:
            _dashed(frame, x1, y1, x2, y2, colour)
            text = f"#{v.identity_id} {v.label} (hidden)"
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            text = f"#{v.identity_id} {v.label} " + ("new" if v.is_new else f"seen before ({v.score:.2f})")
        cv2.putText(frame, text, (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)


def _draw_search(frame, views, selected: int | None, hits) -> None:
    for v in views:
        x1, y1, x2, y2 = (int(c) for c in v.box)
        chosen = v.track_id == selected
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255) if chosen else (160, 160, 160), 3 if chosen else 1)
        cv2.putText(frame, f"{v.label}" + (f" #{v.identity_id}" if v.identity_id is not None else ""),
                    (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255) if chosen else (160, 160, 160), 1)
    y = 60
    cv2.putText(frame, "click an object; most similar stored identities:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for ident, score in hits[:5]:
        y += 22
        cv2.putText(frame, f"  #{ident.id} {ident.label}  cosine {score:.2f}  seen {ident.appearances}x",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _color(ident.id), 1)
    if selected is not None and not hits:
        cv2.putText(frame, "  (nothing stored of this kind yet)", (10, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def _draw_anomaly(frame, views, engine: VisionEngine, baselines: dict) -> None:
    for v in views:
        x1, y1, x2, y2 = (int(c) for c in v.box)
        if v.anomaly is None:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 1)
            cv2.putText(frame, f"{v.label} (learning normal)", (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
            continue
        base = baselines.get(v.label)
        if base is None or len(base) == 0:
            base = engine.anomaly_baseline(v.label)
            baselines[v.label] = base if base is not None else np.empty(0)
        # rank against the label's own normals: 0 = ordinary, 1 = stranger than everything stored
        pct = float((base < v.anomaly).mean()) if len(base) else 0.0
        colour = (0, int(255 * (1 - pct)), int(255 * pct))
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, f"{v.label} unusual {pct:.0%}", (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="camera index, or a video file")
    ap.add_argument("--db", type=Path, default=Path("data/live.db"))
    ap.add_argument("--index", type=Path, default=Path("data/live.faiss"))
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    probe = cv2.VideoCapture(source)
    if not probe.isOpened():
        raise SystemExit(f"cannot open source {args.source!r}")
    fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
    is_file = not isinstance(source, int)
    probe.release()

    verifier, threshold = _fit_verifier(_DEFAULT_CACHE)
    mem_cfg = replace(load_memory_config(), db_path=str(args.db), index_path=str(args.index))
    engine = VisionEngine.from_configs(
        load_detector_config(), load_appearance_config(), mem_cfg, verifier, threshold,
        load_tracker_config(), load_reid_config(), load_video_config(), load_anomaly_config(), fps,
    )
    print(f"source {args.source} at {fps:g} fps | memory {mem_cfg.db_path} ({len(engine.memory)} identities)")
    print("keys: 1 memory   2 search (click an object)   3 anomaly   q quit")

    mailbox = LatestFrame()
    stats: dict = {"pipeline_ms": 0.0, "skipped": 0}
    threading.Thread(target=_capture, args=(source, mailbox, fps if is_file else None), daemon=True).start()
    threading.Thread(target=_worker, args=(engine, mailbox, stats), daemon=True).start()

    mode = "memory"
    selected: int | None = None
    hits: list = []
    baselines: dict = {}
    state = {"click": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    win = "vision memory"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    shown = 0
    t0 = time.perf_counter()
    last_idx = -1
    try:
        while not mailbox.closed:
            frame, idx = mailbox.get()
            if frame is None or idx == last_idx:
                time.sleep(0.002)
                continue
            last_idx = idx
            views = engine.view(idx)

            if state["click"] is not None:
                cx, cy = state["click"]; state["click"] = None
                inside = [v for v in views if v.box[0] <= cx <= v.box[2] and v.box[1] <= cy <= v.box[3]]
                if inside:
                    selected = inside[0].track_id
                    hits = engine.similar(selected, k=5)
            if mode == "search" and selected is not None and idx % 15 == 0:
                hits = engine.similar(selected, k=5)

            if mode == "memory":
                _draw_memory(frame, views)
            elif mode == "search":
                _draw_search(frame, views, selected, hits)
            else:
                if idx % 30 == 0:
                    baselines.clear()
                _draw_anomaly(frame, views, engine, baselines)

            shown += 1
            elapsed = time.perf_counter() - t0
            c = engine.counts
            hud = (f"{mode.upper()}  display {shown / max(elapsed, 1e-6):.1f} fps  pipeline {stats['pipeline_ms']:.0f} ms"
                   f" (skips {stats['skipped']})  objects {len(views)}  known {c.rebound}  new {c.created}")
            cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in MODES:
                mode = MODES[key]
            if args.max_frames is not None and idx >= args.max_frames:
                break
    finally:
        mailbox.closed = True
        cv2.destroyAllWindows()
        engine.close()
        c = engine.counts
        print(f"processed {c.processed} of {last_idx + 1} frames | identified new {c.created}, "
              f"recognized {c.rebound}, taken back {c.taken} | identities in memory {len(engine.memory)}")


if __name__ == "__main__":
    main()
