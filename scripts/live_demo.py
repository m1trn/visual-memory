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


class BoxFollower:
    """Move each box with the pixels inside it between pipeline passes.

    The tracker's velocity projection guesses; this looks. A few feature
    points are picked inside every box when a pass lands, then followed frame
    to frame with Lucas-Kanade optical flow (core OpenCV, ~1 ms per box). The
    box moves by the median displacement of its points, so a stray point on
    the background cannot drag it. Each new pass re-anchors everything.
    """

    _LK = dict(winSize=(21, 21), maxLevel=3,
               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

    def __init__(self) -> None:
        self._grey: np.ndarray | None = None
        self._points: dict[int, np.ndarray] = {}
        self._offset: dict[int, np.ndarray] = {}
        self._anchor: int = -1

    def anchor(self, frame_bgr: np.ndarray, views: list[TrackView], pass_idx: int) -> None:
        """A pass landed: pick fresh points inside every box on this frame."""
        if pass_idx == self._anchor:
            return
        self._anchor = pass_idx
        self._grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self._points, self._offset = {}, {}
        h, w = self._grey.shape
        for v in views:
            x1, y1, x2, y2 = (int(c) for c in v.box)
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            mask = np.zeros_like(self._grey)
            # inner two thirds of the box: feature points on the object, not its edge
            mx, my = (x2 - x1) // 6, (y2 - y1) // 6
            mask[y1 + my:y2 - my, x1 + mx:x2 - mx] = 255
            pts = cv2.goodFeaturesToTrack(self._grey, maxCorners=24, qualityLevel=0.01, minDistance=4, mask=mask)
            if pts is not None and len(pts) >= 4:
                self._points[v.track_id] = pts
                self._offset[v.track_id] = np.zeros(2, np.float32)

    def follow(self, frame_bgr: np.ndarray, views: list[TrackView]) -> list[TrackView]:
        """Advance every anchored box by where its points went in this frame."""
        if self._grey is None or not self._points:
            return views
        grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        out = []
        for v in views:
            pts = self._points.get(v.track_id)
            if pts is None:
                out.append(v)
                continue
            nxt, ok, _ = cv2.calcOpticalFlowPyrLK(self._grey, grey, pts, None, **self._LK)
            good = ok.ravel() == 1
            if good.sum() >= 4:
                shift = np.median((nxt[good] - pts[good]).reshape(-1, 2), axis=0)
                self._points[v.track_id] = nxt[good]
                self._offset[v.track_id] = self._offset[v.track_id] + shift.astype(np.float32)
            dx, dy = self._offset[v.track_id]
            out.append(TrackView(v.track_id, v.box + np.array([dx, dy, dx, dy], np.float32), v.label,
                                 v.identity_id, v.is_new, v.score, v.hidden, v.anomaly))
        self._grey = grey
        return out


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
        stats["anchor"] = (idx, frame)      # the display re-anchors its followers on this frame
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
    ap.add_argument("--between", choices=("follow", "predict", "hold"), default="follow",
                    help="how boxes move between pipeline passes: follow the pixels (optical flow), "
                         "predict from the tracker's velocity, or hold still")
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
    follower = BoxFollower()
    try:
        while not mailbox.closed:
            frame, idx = mailbox.get()
            if frame is None or idx == last_idx:
                time.sleep(0.002)
                continue
            last_idx = idx
            if args.between == "hold":
                views = engine.view(engine._last_frame_idx if engine._last_frame_idx >= 0 else idx)
            elif args.between == "predict":
                views = engine.view(idx)
            else:
                anchor = stats.get("anchor")
                base = engine.view(engine._last_frame_idx if engine._last_frame_idx >= 0 else idx)
                if anchor is not None:
                    follower.anchor(anchor[1], base, anchor[0])
                views = follower.follow(frame, base)

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
        elapsed = max(time.perf_counter() - t0, 1e-6)
        print(f"processed {c.processed} of {last_idx + 1} frames | identified new {c.created}, "
              f"recognized {c.rebound}, taken back {c.taken} | identities in memory {len(engine.memory)}")
        print(f"display {shown / elapsed:.1f} fps | pipeline {stats['pipeline_ms']:.0f} ms per pass, "
              f"{stats['skipped']} frames skipped per pass at the end")


if __name__ == "__main__":
    main()
