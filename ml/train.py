"""Fine-tune YOLO11m for human detection in dark / camouflage / forest scenes.

    python train.py              # ~30 epochs on the GPU
    python train.py --epochs 50

The 80-class COCO head is kept (person = class 0) and the backbone is partly frozen so
the model keeps its general knowledge. Best weights: runs/person_ft/weights/best.pt
Use them with:  $env:MODEL_NAME="runs/person_ft/weights/best.pt"
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--base", default="yolo11m.pt")
    ap.add_argument("--name", default="person_ft")
    args = ap.parse_args()

    model = YOLO(args.base)
    model.train(
        data="train_data/data.yaml",
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        device=0,
        workers=4,
        freeze=10,        # keep the early backbone layers
        lr0=0.002,
        patience=10,
        hsv_v=0.6,        # strong brightness jitter -> dark scenes
        project="runs",
        name=args.name,
        exist_ok=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
