"""Fetch one VisDrone-MOT scene as a MOT-format sequence, for non-person evaluation.

Every labelled number in this project is MOT17 pedestrians. The object branch
of the appearance router - DINOv2 plus colour bands for anything that is not a
person - has never been scored on video with ground truth. VisDrone-MOT is
drone footage with persistent track ids across ten classes including car, van,
truck, bus and motor, so it can ask the same questions of vehicles that MOT17
asks of people.

The Voxel51 mirror stores frames as individual JPEGs and labels as one
FiftyOne export (`samples.json`), where each detection carries `index` - the
track id - and `scene_id` names the clip. This writes the subset for one scene
in the layout `motchallenge.py` already reads: seqinfo.ini, img1/, gt/gt.txt.

Class ids follow the VisDrone convention and are recorded in classes.txt beside
the sequence, since MOT17's gt uses 1 for pedestrian and the readers filter on
it; `mot_eval.py --classes` selects which ones to score.

Usage: python scripts/fetch_visdrone.py [--scene uav0000086_00000_v] [--frames 400]
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

REPO = "Voxel51/visdrone-mot"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
# VisDrone's own ordering; 1 is what MOT17 calls a pedestrian, so the readers
# keep working unchanged and vehicles sit above it.
CLASSES = ["ignored", "pedestrian", "people", "bicycle", "car", "van", "truck",
           "tricycle", "awning-tricycle", "bus", "motor", "others"]


def _get(url: str, into: Path) -> bool:
    if into.exists() and into.stat().st_size > 0:
        return True
    into.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"  failed {url}: {exc}")
        return False
    into.write_bytes(data)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=None, help="scene_id to fetch; omitted lists the vehicle-heaviest")
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--dest", type=Path, default=Path("data/visdrone"))
    ap.add_argument("--labels", type=Path, default=Path("data/visdrone/samples.json"))
    args = ap.parse_args()

    if not _get(f"{BASE}/samples.json", args.labels):
        raise SystemExit("could not fetch labels")
    samples = json.loads(args.labels.read_text(encoding="utf-8"))["samples"]

    by_scene: dict[str, list] = defaultdict(list)
    for s in samples:
        by_scene[s["scene_id"]].append(s)

    if args.scene is None:
        vehicles = {"car", "van", "truck", "bus", "motor"}
        rank = []
        for scene, rows in by_scene.items():
            counts = Counter(d["label"] for r in rows for d in (r.get("detections") or []))
            v = sum(counts[c] for c in vehicles)
            rank.append((v / max(sum(counts.values()), 1), v, len(rows), scene, counts.most_common(3)))
        rank.sort(reverse=True)
        print(f"{'vehicle share':>14}{'boxes':>8}{'frames':>8}  scene / top labels")
        for share, v, n, scene, top in rank[:10]:
            print(f"{share:>13.0%}{v:>8}{n:>8}  {scene}  {top}")
        print("\nre-run with --scene <id>")
        return

    rows = sorted(by_scene[args.scene], key=lambda r: r["frame_number"])[: args.frames]
    if not rows:
        raise SystemExit(f"no frames for scene {args.scene!r}")
    seq = args.dest / args.scene
    lines = []
    for n, row in enumerate(rows, start=1):
        name = Path(row["filepath"]).name
        if not _get(f"{BASE}/data/{name}", seq / "img1" / f"{n:06d}.jpg"):
            break
        for d in row.get("detections") or []:
            bx, by, bw, bh = d["bounding_box"]      # fractions of the frame
            cls = CLASSES.index(d["label"]) if d["label"] in CLASSES else 0
            lines.append((n, int(d["index"]), bx, by, bw, bh, 1 - int(d.get("occlusion", 0) == 2),
                          cls, 1.0 - 0.5 * int(d.get("occlusion", 0) == 1)))
        if n % 25 == 0:
            print(f"  {n}/{len(rows)}", flush=True)

    frames_got = len(list((seq / "img1").glob("*.jpg")))
    import cv2

    first = cv2.imread(str(seq / "img1" / "000001.jpg"))
    h, w = first.shape[:2]
    (seq / "seqinfo.ini").write_text(
        f"[Sequence]\nname={args.scene}\nimDir=img1\nframeRate=25\n"
        f"seqLength={frames_got}\nimWidth={w}\nimHeight={h}\nimExt=.jpg\n", encoding="utf-8")
    (seq / "gt").mkdir(exist_ok=True)
    (seq / "gt" / "gt.txt").write_text("\n".join(
        f"{n},{i},{bx * w:.1f},{by * h:.1f},{bw * w:.1f},{bh * h:.1f},{keep},{cls},{vis:.1f}"
        for n, i, bx, by, bw, bh, keep, cls, vis in lines if n <= frames_got) + "\n", encoding="utf-8")
    (seq / "classes.txt").write_text("\n".join(f"{i} {c}" for i, c in enumerate(CLASSES)), encoding="utf-8")
    kinds = Counter(CLASSES[c] for *_, c, _ in lines)
    print(f"\n{args.scene}: {frames_got} frames at {w}x{h}, {len(lines)} boxes")
    print("  labels:", dict(kinds.most_common()))
    print(f"  -> {seq}")


if __name__ == "__main__":
    main()
