"""Build the fine-tuning dataset in YOLO format -> train_data/

    python -I build_training_set.py

Sources (all single-class "person"; class id 0 = COCO's person id, so the pretrained
80-class head is kept):
  * raw/soldier_b, raw/soldier_a, raw/camouflage  (Roboflow, CC BY 4.0): train+valid used
    for training; their TEST splits are copied to camo_test/ and never trained on.
  * COCO val2017 people (boxes from instances_val2017.json) + COCO images WITHOUT people
    (empty labels = background) so false alarms do not go up.
    Images already used in data/ (the regression test set) are excluded.
Excluded on purpose: raw/hidden (fire/smoke scenes, "Private" license).
"""
import json
import random
import shutil
import zipfile
from pathlib import Path

random.seed(7)
ROOT = Path(".")
OUT = ROOT / "train_data"
DL = Path.home() / "Downloads"
N_COCO_PERSON, N_COCO_NEG = 10**6, 10**6  # use every COCO val image not already in data/
VAL_FRAC = 0.12

for p in (OUT, ROOT / "camo_test"):
    if p.exists():
        shutil.rmtree(p)
for split in ("train", "val"):
    (OUT / split / "images").mkdir(parents=True)
    (OUT / split / "labels").mkdir(parents=True)
for sub in ("images", "labels"):
    (ROOT / "camo_test" / sub).mkdir(parents=True)


def copy_roboflow(name):
    n = 0
    for split, dest in (("train", "train"), ("valid", "val")):
        for img in (ROOT / "raw" / name / split / "images").glob("*"):
            lab = ROOT / "raw" / name / split / "labels" / (img.stem + ".txt")
            if not lab.exists():
                continue
            # keep only class 0 boxes (single-class datasets)
            lines = [l for l in lab.read_text().splitlines() if l.split()[:1] == ["0"]]
            if not lines:
                continue
            shutil.copy(img, OUT / dest / "images" / f"{name}_{img.name}")
            (OUT / dest / "labels" / f"{name}_{img.stem}.txt").write_text("\n".join(lines))
            n += 1
    for img in (ROOT / "raw" / name / "test" / "images").glob("*"):
        lab = ROOT / "raw" / name / "test" / "labels" / (img.stem + ".txt")
        if lab.exists():
            shutil.copy(img, ROOT / "camo_test" / "images" / f"{name}_{img.name}")
            shutil.copy(lab, ROOT / "camo_test" / "labels" / f"{name}_{img.stem}.txt")
    return n


counts = {n: copy_roboflow(n) for n in ("soldier_b", "soldier_a", "camouflage")}
print("Roboflow images into train/val:", counts)


def copy_hidden():
    """raw/hidden: class 0 = human, 1 = fire, 2 = smoke. Keep only human boxes; images with no
    human become background (empty label). All three splits are used for training/validation."""
    n_h = n_bg = 0
    for split in ("train", "valid", "test"):
        dest = "val" if split == "valid" else "train"
        for img in (ROOT / "raw" / "hidden" / split / "images").glob("*"):
            lab = ROOT / "raw" / "hidden" / split / "labels" / (img.stem + ".txt")
            lines = [l for l in (lab.read_text().splitlines() if lab.exists() else []) if l.split()[:1] == ["0"]]
            shutil.copy(img, OUT / dest / "images" / f"hidden_{split}_{img.name}")
            (OUT / dest / "labels" / f"hidden_{split}_{img.stem}.txt").write_text("\n".join(lines))
            n_h += bool(lines)
            n_bg += not lines
    return n_h, n_bg


print("hidden (images with humans, background-only images):", copy_hidden())

# ---------------------------------------------------------------- COCO
used = {p.name.replace("coco_", "") for sub in ("person", "no_person") for p in (ROOT / "data" / sub).glob("coco_*")}
coco = json.loads(zipfile.ZipFile(DL / "annotations_trainval2017.zip").read("annotations/instances_val2017.json"))
cats = {c["id"]: c["name"] for c in coco["categories"]}
imgs = {i["id"]: i for i in coco["images"]}
boxes, has_person = {}, set()
for a in coco["annotations"]:
    if cats[a["category_id"]] == "person":
        has_person.add(a["image_id"])
        if not a["iscrowd"]:
            boxes.setdefault(a["image_id"], []).append(a["bbox"])

cand_p = [i for i in boxes if imgs[i]["file_name"] not in used]
cand_n = [i for i in imgs if i not in has_person and imgs[i]["file_name"] not in used]
pick = [(i, True) for i in random.sample(cand_p, min(N_COCO_PERSON, len(cand_p)))]
pick += [(i, False) for i in random.sample(cand_n, min(N_COCO_NEG, len(cand_n)))]

val = zipfile.ZipFile(DL / "val2017.zip")
for i, positive in pick:
    info = imgs[i]
    split = "val" if random.random() < VAL_FRAC else "train"
    (OUT / split / "images" / f"coco_{info['file_name']}").write_bytes(val.read(f"val2017/{info['file_name']}"))
    lines = []
    if positive:
        W, H = info["width"], info["height"]
        for x, y, w, h in boxes[i]:
            if w < 4 or h < 4:
                continue
            lines.append(f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
    (OUT / split / "labels" / f"coco_{Path(info['file_name']).stem}.txt").write_text("\n".join(lines))
print(f"COCO: {len(pick)} images ({N_COCO_PERSON} with people, {N_COCO_NEG} background)")

# --------------------------------------------------------------- yaml
names = [c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])]
# YOLO's 80-class order is the COCO category order with ids compressed; person must be index 0
coco80 = ["person"] + [n for n in names if n != "person"]
(OUT / "data.yaml").write_text(
    f"path: {OUT.resolve().as_posix()}\ntrain: train/images\nval: val/images\nnames:\n"
    + "".join(f"  {i}: {n}\n" for i, n in enumerate(coco80))
)
for s in ("train", "val"):
    print(s, len(list((OUT / s / "images").iterdir())), "images")
print("camo_test:", len(list((ROOT / "camo_test" / "images").iterdir())), "images (held out)")
