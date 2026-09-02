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

The result is not assumed to be an improvement. Training uses one sequence and
validation another, and the honest test is the project's own end-to-end
evaluation on the held-out sequence afterwards. A model that memorises one
camera can score well here and worse in `mot_eval.py`, which is the outcome
this split exists to expose.
"""

from __future__ import annotations

import argparse
from pathlib import Path


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
    ap.add_argument("--out", type=Path, default=Path("data/models/yolo11n_mot_768.onnx"))
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        freeze=args.freeze, device="cpu", workers=0, project=str(Path("data/detector/runs").resolve()),
        name=args.name, exist_ok=True,
        # A fixed overhead camera: flipping is fine, rotating and shearing are
        # not - they produce viewpoints this camera can never see.
        degrees=0.0, shear=0.0, perspective=0.0, mosaic=0.5, fliplr=0.5,
        patience=10, seed=0, val=True, plots=False,
    )
    exported = model.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(exported).replace(args.out)
    print(f"\n-> {args.out}")
    print("Now score it the same way as everything else:")
    print(f"  set detector.model_path to {args.out} in configs/default.yaml, then")
    print("  python scripts/reid_sweep.py --data data/mot/MOT17-09-FRCNN --rebuild")


if __name__ == "__main__":
    main()
