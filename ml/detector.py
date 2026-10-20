"""Human verification for the intrusion detection system.

Public API:
    verify_person(image) -> dict
        image: file path, JPEG/PNG bytes, PIL image or BGR numpy array.

Accuracy measures beyond a plain "is there a person box":
  * a stronger free model (YOLO11m by default; set MODEL_NAME for l/x or RT-DETR)
  * larger inference size so small/far people are still found
  * person-class only, with a confidence threshold AND a minimum box size
  * box-shape sanity check (rejects very wide/flat boxes that are rarely people)
  * optional test-time augmentation (TTA) for hard images
  * optional low-light enhancement (CLAHE) for dark ESP32-CAM frames
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------- settings
# fine-tuned YOLO11m (dark/camouflage/forest); set MODEL_NAME=yolo11m.pt for the plain pretrained model
MODEL_NAME = os.getenv("MODEL_NAME", str(Path(__file__).parent / "models" / "person_ft.pt"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.50"))  # decision threshold
IMG_SIZE = int(os.getenv("IMG_SIZE", "960"))
MIN_BOX_AREA = float(os.getenv("MIN_BOX_AREA", "0.005"))  # fraction of image area
MIN_BOX_HEIGHT = float(os.getenv("MIN_BOX_HEIGHT", "0.10"))  # fraction of image height
MAX_ASPECT = float(os.getenv("MAX_ASPECT", "3.0"))  # width/height; loose enough for a lying/crawling person
USE_TTA = os.getenv("USE_TTA", "0") == "1"
ENHANCE_DARK = os.getenv("ENHANCE_DARK", "1") == "1"
ZOOM_OUT_PASS = os.getenv("ZOOM_OUT_PASS", "1") == "1"
ZOOM_PAD = float(os.getenv("ZOOM_PAD", "0.5"))  # border added per side, fraction of long edge
DARK_MEAN = 70  # mean gray level under which a frame counts as "dark"
PERSON_CLASS = 0  # COCO

_model = None
_device = None


def _pick_device():
    import torch

    return 0 if torch.cuda.is_available() else "cpu"


def get_model():
    """Load the model once (first call downloads weights if missing)."""
    global _model, _device
    if _model is None:
        from ultralytics import RTDETR, YOLO

        _device = _pick_device()
        cls = RTDETR if "rtdetr" in MODEL_NAME.lower() else YOLO
        _model = cls(MODEL_NAME)
        # warm up so the first real request is not slow
        _model.predict(np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8), verbose=False, device=_device)
    return _model


# ------------------------------------------------------------------ input
def load_image(image) -> np.ndarray:
    """Return a BGR uint8 numpy image from path / bytes / PIL / ndarray."""
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, (str, Path)):
        data = np.fromfile(str(image), dtype=np.uint8)  # works with unicode paths on Windows
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif isinstance(image, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    else:  # PIL
        img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def enhance_if_dark(img: np.ndarray) -> np.ndarray:
    """CLAHE on the luminance channel for dark frames (night / poor ESP32-CAM exposure)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.mean() >= DARK_MEAN:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


# -------------------------------------------------------------- detection
def detect(image, conf_floor: float = 0.25, tta: bool | None = None):
    """Return (all raw person detections, ms, annotated-source image).

    Each detection: {"confidence", "box": [x1,y1,x2,y2], "area", "height", "aspect"}
    where area/height are fractions of the image. No threshold filtering here,
    only the low `conf_floor`, so evaluate.py can sweep thresholds cheaply.
    """
    img = load_image(image)
    if ENHANCE_DARK:
        img = enhance_if_dark(img)
    h, w = img.shape[:2]

    t0 = time.perf_counter()
    dets = _infer(img, conf_floor, USE_TTA if tta is None else tta, pad=0)
    # Zoom-out pass: a person very close to the camera fills the frame and YOLO
    # (trained mostly on full-body views) scores them low. Padding the image
    # makes them look like a normal whole-body subject. Only run when needed.
    if ZOOM_OUT_PASS and not any(is_valid(d) for d in dets):
        dets += _infer(img, conf_floor, False, pad=int(max(h, w) * ZOOM_PAD))
    ms = (time.perf_counter() - t0) * 1000
    return dets, ms, img


def _infer(img, conf_floor, tta, pad):
    h, w = img.shape[:2]
    src = img
    if pad:
        src = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    res = get_model().predict(
        src,
        classes=[PERSON_CLASS],
        conf=conf_floor,
        imgsz=IMG_SIZE,
        augment=tta,
        device=_device,
        verbose=False,
    )[0]
    dets = []
    for b in res.boxes:
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
        # map back to original image coordinates and clip to the frame
        x1, x2 = min(max(x1 - pad, 0), w), min(max(x2 - pad, 0), w)
        y1, y2 = min(max(y1 - pad, 0), h), min(max(y2 - pad, 0), h)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 1 or bh <= 1:
            continue
        dets.append(
            {
                "confidence": float(b.conf[0]),
                "box": [round(x1), round(y1), round(x2), round(y2)],
                "area": (bw * bh) / (w * h),
                "height": bh / h,
                "aspect": bw / bh,
                "zoom_out": bool(pad),
            }
        )
    return dets


def is_valid(det: dict, conf=None, min_area=None, min_height=None, max_aspect=None) -> bool:
    return (
        det["confidence"] >= (CONF_THRESHOLD if conf is None else conf)
        and det["area"] >= (MIN_BOX_AREA if min_area is None else min_area)
        and det["height"] >= (MIN_BOX_HEIGHT if min_height is None else min_height)
        and det["aspect"] <= (MAX_ASPECT if max_aspect is None else max_aspect)
    )


def _iou_or_contained(a, b) -> float:
    """Overlap of two boxes relative to the smaller one (catches a box nested inside another)."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0


def _dedupe(dets: list, thr: float = 0.7) -> list:
    """Keep one box per person: the normal and zoom-out passes can both fire on the same person."""
    kept = []
    for d in dets:  # already sorted by confidence, best first
        if all(_iou_or_contained(d["box"], k["box"]) < thr for k in kept):
            kept.append(d)
    return kept


def verify_person(image, conf: float | None = None) -> dict:
    """Main entry point. Returns:
    {"person": bool, "confidence": float, "count": int, "boxes": [...], "ms": float}
    """
    dets, ms, _ = detect(image)
    good = [d for d in dets if is_valid(d, conf=conf)]
    good.sort(key=lambda d: d["confidence"], reverse=True)
    good = _dedupe(good)
    return {
        "person": bool(good),
        "confidence": round(good[0]["confidence"], 4) if good else 0.0,
        "count": len(good),
        "boxes": [{"box": d["box"], "confidence": round(d["confidence"], 4)} for d in good],
        "ms": round(ms, 1),
    }


def annotate(img: np.ndarray, result: dict) -> np.ndarray:
    out = img.copy()
    for b in result["boxes"]:
        x1, y1, x2, y2 = b["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # put the label inside the box so it never collides with the status text
        cv2.putText(out, f"person {b['confidence']:.2f}", (x1 + 6, min(max(y1 + 24, 60), out.shape[0] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    label = "HUMAN DETECTED" if result["person"] else "NO HUMAN"
    color = (0, 0, 255) if result["person"] else (0, 160, 0)
    cv2.putText(out, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return out


if __name__ == "__main__":
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    r = verify_person(path)
    print(json.dumps(r, indent=2))
    print("HUMAN INTRUSION DETECTED" if r["person"] else "NO HUMAN DETECTED")
    if "--show" in sys.argv:
        cv2.imshow("result", annotate(load_image(path), r))
        cv2.waitKey(0)
