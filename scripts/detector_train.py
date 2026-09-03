r"""Fine-tune the detector on labelled pedestrians, then export it as ONNX.

Runs in a THROWAWAY environment, not the project's own: `ultralytics` is AGPL
and is never imported by the shipped system, which runs the exported graph
through onnxruntime with its own decoding. This script exists to produce a
.onnx file that drops into data/models/ with no code change.

    python -m venv .venv-train
    .venv-train\Scripts\python -m pip install ultralytics
    .venv-train\Scripts\python scripts/detector_train.py --epochs 40

Why fine-tune at all: the shipped weights are stock COCO - eighty everyday
classes from ordinary photographs - while this footage is small, distant,
overhead pedestrians, and the detector finds 69% of them. Everything
downstream is capped by that number.

The result is not assumed to be an improvement. Training uses one set of
sequences and validation another, and the honest test is the project's own
end-to-end evaluation on the held-out sequence afterwards. A model that
memorises its training cameras can score well here and worse in `mot_eval.py`,
which is the outcome this split exists to expose.

Two settings exist because of what a first attempt measured. Selecting on mAP
chose a model that made 31% fewer detections than stock: precision, mAP and
MOTA all rose while identity switches went 9 to 37, because fewer detections
means gaps in tracks and every restart is a new identity. What this system
needs from a detector is RECALL, so `--select recall` reports the epoch that
found the most people rather than the one that scored its finds best. Mosaic
augmentation is off by default for the same reason: pasting four frames
together teaches robustness to composition that surveillance footage never
varies, at the cost of the small distant people that are the actual difficulty.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _report_best(run: Path, select: str) -> None:
    """Name the epoch that won on the chosen measure, and on the other one.

    ultralytics saves best.pt on its own fitness score, which is dominated by
    mAP. Printing both makes it visible when the two disagree - which is
    exactly the case that produced a model scoring better on every metric that
    counts false positives and worse on the only one that counts continuity.
    """
    results = run / "results.csv"
    if not results.exists():
        return
    rows = [r.split(",") for r in results.read_text().strip().splitlines()[1:]]
    if not rows:
        return
    seen: dict[str, list[str]] = {}
    for r in rows:
        seen[r[0]] = r                      # a resumed run repeats an epoch; keep the last
    epochs = list(seen.values())
    by_recall = max(epochs, key=lambda r: float(r[6]))
    by_map = max(epochs, key=lambda r: float(r[7]))
    print()
    print(f"  best recall  epoch {by_recall[0]:>3}: R {float(by_recall[6]):.3f}  mAP50 {float(by_recall[7]):.3f}")
    print(f"  best mAP50   epoch {by_map[0]:>3}: R {float(by_map[6]):.3f}  mAP50 {float(by_map[7]):.3f}")
    if by_recall[0] != by_map[0]:
        print(f"  they disagree. best.pt holds the mAP winner; --select {select} says "
              f"epoch {by_recall[0] if select == 'recall' else by_map[0]} is the one to test.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/detector/dataset.yaml"))
    ap.add_argument("--weights", default="yolo11n.pt", help="starting point; not trained from scratch")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=768, help="match the size the export will use")
    ap.add_argument("--batch", type=int, default=4, help="small, because this trains on a CPU")
    ap.add_argument("--freeze", type=int, default=10,
                    help="backbone layers held fixed; the features are already good, "
                         "it is the detection head that has not seen this domain")
    ap.add_argument("--name", default="mot_person")
    ap.add_argument("--mosaic", type=float, default=0.0,
                    help="mosaic augmentation; off by default, since pasting four frames "
                         "together varies a composition that a fixed camera never does")
    ap.add_argument("--select", choices=("map", "recall"), default="recall",
                    help="which epoch to report as best. ultralytics always saves best.pt "
                         "on its own fitness score; this reports the recall-best epoch too, "
                         "because recall is what this system is short of")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run from its last checkpoint; "
                         "training on a CPU takes hours and is worth restarting rather than repeating")
    ap.add_argument("--out", type=Path, default=Path("data/models/yolo11n_mot_768.onnx"))
    args = ap.parse_args()

    from ultralytics import YOLO

    run = Path("data/detector/runs") / args.name
    last = run / "weights" / "last.pt"
    if args.resume and last.exists():
        print(f"resuming from {last}")
        YOLO(str(last)).train(resume=True)
        model = YOLO(str(run / "weights" / "best.pt"))
        exported = model.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=False)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        Path(exported).replace(args.out)
        print(f"\n-> {args.out}")
        return

    model = YOLO(args.weights)
    model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        freeze=args.freeze, device="cpu", workers=0, project=str(Path("data/detector/runs").resolve()),
        name=args.name, exist_ok=True,
        # A fixed overhead camera: flipping is fine, rotating and shearing are
        # not - they produce viewpoints this camera can never see.
        degrees=0.0, shear=0.0, perspective=0.0, mosaic=args.mosaic, fliplr=0.5,
        patience=10, seed=0, val=True, plots=False,
    )
    _report_best(run, args.select)
    exported = model.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(exported).replace(args.out)
    print(f"\n-> {args.out}")
    print("Now score it the same way as everything else:")
    print(f"  set detector.model_path to {args.out} in configs/default.yaml, then")
    print("  python scripts/reid_sweep.py --data data/mot/MOT17-09-FRCNN --rebuild")


if __name__ == "__main__":
    main()
