"""Measure accuracy on a labeled folder set and pick the best threshold.

Layout:
    data/person/*.jpg       images that contain at least one person
    data/no_person/*.jpg    dogs, cats, empty rooms, objects, shadows, ...

Usage:
    python evaluate.py                 # evaluate ./data
    python evaluate.py --data other    # another folder with the same layout
    python evaluate.py --tta           # with test-time augmentation

Writes results/report.txt, results/predictions.csv and results/threshold_sweep.png
"""
import argparse
import csv
from pathlib import Path

import detector

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTS)


def metrics(rows, thr):
    tp = fp = fn = tn = 0
    for r in rows:
        pred = any(detector.is_valid(d, conf=thr) for d in r["dets"])
        if r["label"] and pred:
            tp += 1
        elif r["label"] and not pred:
            fn += 1
        elif not r["label"] and pred:
            fp += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    far = fp / (fp + tn) if fp + tn else 0.0
    acc = (tp + tn) / max(len(rows), 1)
    return dict(thr=thr, tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1, far=far, acc=acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    root = Path(args.data)
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    rows = []
    for label, sub in ((True, "person"), (False, "no_person")):
        for p in collect(root / sub):
            dets, ms, _ = detector.detect(p, tta=args.tta)
            rows.append({"path": p, "label": label, "dets": dets, "ms": ms})
            print(f"{'P ' if label else 'NP'} {p.name:40s} "
                  f"best={max((d['confidence'] for d in dets), default=0):.2f} {ms:.0f}ms")
    if not rows:
        raise SystemExit(f"No images found in {root}/person and {root}/no_person")

    sweep = [metrics(rows, round(t, 2)) for t in [x / 100 for x in range(20, 91, 5)]]
    # best = max recall subject to false-alarm rate <= 5%, ties broken by F1
    ok = [m for m in sweep if m["far"] <= 0.05] or sweep
    best = max(ok, key=lambda m: (m["recall"], m["f1"]))
    cur = metrics(rows, detector.CONF_THRESHOLD)

    lines = [
        f"Model: {detector.MODEL_NAME}  imgsz={detector.IMG_SIZE}  TTA={args.tta}",
        f"Images: {sum(r['label'] for r in rows)} person, {sum(not r['label'] for r in rows)} no_person",
        f"Mean inference: {sum(r['ms'] for r in rows) / len(rows):.0f} ms",
        "",
        "thr   TP  FP  FN  TN  precision recall   F1    false-alarm-rate",
    ]
    for m in sweep:
        lines.append(f"{m['thr']:.2f} {m['tp']:4d}{m['fp']:4d}{m['fn']:4d}{m['tn']:4d}  "
                     f"{m['precision']:.3f}     {m['recall']:.3f}  {m['f1']:.3f}  {m['far']:.3f}")
    lines += [
        "",
        f"Current CONF_THRESHOLD={detector.CONF_THRESHOLD}: precision={cur['precision']:.3f} "
        f"recall={cur['recall']:.3f} false-alarm-rate={cur['far']:.3f}",
        f"Suggested threshold: {best['thr']} (recall {best['recall']:.3f}, false-alarm {best['far']:.3f})",
        "",
        "Mistakes at the current threshold:",
    ]
    for r in rows:
        pred = any(detector.is_valid(d) for d in r["dets"])
        if pred != r["label"]:
            kind = "MISSED PERSON (FN)" if r["label"] else "FALSE ALARM (FP)"
            lines.append(f"  {kind}: {r['path']}")
    report = "\n".join(lines)
    print("\n" + report)
    (out / "report.txt").write_text(report, encoding="utf-8")

    with open(out / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label_person", "best_confidence", "predicted_person", "ms"])
        for r in rows:
            best_c = max((d["confidence"] for d in r["dets"] if detector.is_valid(d, conf=0)), default=0)
            w.writerow([r["path"], r["label"], f"{best_c:.3f}",
                        any(detector.is_valid(d) for d in r["dets"]), f"{r['ms']:.0f}"])

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [m["thr"] for m in sweep]
        plt.plot(xs, [m["precision"] for m in sweep], label="precision")
        plt.plot(xs, [m["recall"] for m in sweep], label="recall")
        plt.plot(xs, [m["far"] for m in sweep], label="false-alarm rate")
        plt.xlabel("confidence threshold")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(out / "threshold_sweep.png", dpi=150)
    except ImportError:
        print("(install matplotlib to also get threshold_sweep.png)")


if __name__ == "__main__":
    main()
