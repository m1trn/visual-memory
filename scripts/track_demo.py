"""Phase 5 proof: run detector + ByteTracker on a video, save an annotated copy.

Usage: python scripts/track_demo.py [--video PATH] [--max-frames N]
(default video: data/samples/vtest.avi, downloaded if missing)
"""

from __future__ import annotations

import argparse
from collections import Counter
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vision_memory.config import load_detector_config, load_tracker_config, load_video_config  # noqa: E402
from vision_memory.detector import YoloOnnxDetector  # noqa: E402
from vision_memory.tracker import ByteTracker  # noqa: E402

_DEFAULT_VIDEO = Path("data/samples/vtest.avi")
_DEFAULT_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi"


def _color(track_id: int) -> tuple[int, int, int]:
    """Deterministic BGR color from a track id."""
    r = (37 * track_id) % 256
    g = (17 + 91 * track_id) % 256
    b = (211 + 53 * track_id) % 256
    return int(b), int(g), int(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=_DEFAULT_VIDEO)
    ap.add_argument("--max-frames", type=int, default=300)
    args = ap.parse_args()
    if not args.video.exists():
        args.video.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_DEFAULT_URL, args.video)

    detector = YoloOnnxDetector(load_detector_config())
    tracker_cfg = load_tracker_config()
    tracker = ByteTracker(tracker_cfg)
    video_cfg = load_video_config()

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dir = Path("data/tracking")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.video.stem}_tracked.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    det_times: list[float] = []
    trk_times: list[float] = []
    confirmed: dict[int, str] = {}
    stale_skipped = 0
    assigned = 0
    high_conf_dets = 0
    lost_count = 0
    frame_idx = 0
    t_start = time.perf_counter()

    while frame_idx < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        detections = None
        if frame_idx % video_cfg.detect_every_n_frames == 0:
            t0 = time.perf_counter()
            detections = detector.detect(frame)
            det_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        active = tracker.update(detections)
        trk_times.append(time.perf_counter() - t0)
        lost_count += len(tracker.pop_lost())
        if detections is not None:
            high_conf_dets += sum(1 for d in detections if d.score >= tracker_cfg.high_conf)
            assigned += sum(1 for t in active
                            if t.time_since_update == 0 and t.score >= tracker_cfg.high_conf)

        for trk in active:
            # A track that has missed a whole detector cycle is coasting on
            # prediction alone; drawing it puts a box where nothing was seen.
            if trk.time_since_update > video_cfg.detect_every_n_frames:
                stale_skipped += 1
                continue
            confirmed[trk.id] = trk.label
            x1, y1, x2, y2 = (int(v) for v in trk.box)
            color = _color(trk.id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"#{trk.id} {trk.label}", (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        writer.write(frame)
        frame_idx += 1

    wall = time.perf_counter() - t_start
    cap.release()
    writer.release()

    print(f"{args.video.name}: {frame_idx} frames in {wall:.1f}s ({frame_idx / wall:.1f} fps)")
    if det_times:
        print(f"  detector: {statistics.median(det_times) * 1000:.1f} ms/frame (median, {len(det_times)} runs)")
    if trk_times:
        print(f"  tracker:  {statistics.median(trk_times) * 1000:.1f} ms/frame (median, {len(trk_times)} runs)")
    by_label = Counter(confirmed.values())
    print(f"  confirmed identities: {len(confirmed)} " +
          "(" + ", ".join(f"{n} {lbl}" for lbl, n in by_label.most_common()) + ")")
    print(f"  stale boxes not drawn: {stale_skipped}")
    # Identity count alone rewards a tracker that merges everything, so it is
    # only meaningful next to how much of the footage stayed covered.
    print(f"  detections assigned: {assigned}/{high_conf_dets} "
          f"({assigned / max(high_conf_dets, 1):.1%} coverage)")
    print(f"  tracks spawned: {tracker._next_id - 1}")
    print(f"  lost tracks: {lost_count}")
    print(f"annotated -> {out_path}")


if __name__ == "__main__":
    main()
