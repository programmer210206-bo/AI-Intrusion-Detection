"""Build data/person and data/no_person from the downloaded COCO val2017 and ExDark zips.

    python -I build_dataset.py [downloads_dir]

COCO is labeled from its annotations. For person images we keep a mix of hard cases
(small people) and ordinary ones; non-person images favour things that cause false
alarms (animals, chairs, bags). ExDark gives dark images; its non-person folders are
NOT guaranteed person-free, so review any false alarm that comes from an exdark_ file.
"""
import json
import random
import sys
import zipfile
from pathlib import Path

DL = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads"
OUT = Path("data")
random.seed(42)

N_COCO_PERSON_SMALL, N_COCO_PERSON_MIXED = 30, 30
N_COCO_NEG_HARD, N_COCO_NEG_RANDOM = 35, 25
N_EX_PERSON, N_EX_NEG_EACH = 60, 6
HARD_NEG = {"cat", "dog", "bird", "horse", "sheep", "cow", "bear", "chair", "backpack", "handbag",
            "suitcase", "potted plant", "couch", "teddy bear"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

for sub in ("person", "no_person"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- COCO
ann_zip = zipfile.ZipFile(DL / "annotations_trainval2017.zip")
coco = json.loads(ann_zip.read("annotations/instances_val2017.json"))
cats = {c["id"]: c["name"] for c in coco["categories"]}
imgs = {i["id"]: i for i in coco["images"]}
person_h, cat_names = {}, {}
for a in coco["annotations"]:
    i = a["image_id"]
    cat_names.setdefault(i, set()).add(cats[a["category_id"]])
    if cats[a["category_id"]] == "person" and not a["iscrowd"]:
        person_h[i] = max(person_h.get(i, 0), a["bbox"][3] / imgs[i]["height"])

small = [i for i, h in person_h.items() if 0.12 <= h <= 0.35]
others = [i for i, h in person_h.items() if h > 0.35]
no_person = [i for i in imgs if "person" not in cat_names.get(i, set())]
hard_neg = [i for i in no_person if cat_names.get(i, set()) & HARD_NEG]
rand_neg = [i for i in no_person if i not in set(hard_neg)]

picks = [("person", i) for i in random.sample(small, N_COCO_PERSON_SMALL)]
picks += [("person", i) for i in random.sample(others, N_COCO_PERSON_MIXED)]
picks += [("no_person", i) for i in random.sample(hard_neg, N_COCO_NEG_HARD)]
picks += [("no_person", i) for i in random.sample(rand_neg, N_COCO_NEG_RANDOM)]

val = zipfile.ZipFile(DL / "val2017.zip")
for sub, i in picks:
    name = imgs[i]["file_name"]
    (OUT / sub / f"coco_{name}").write_bytes(val.read(f"val2017/{name}"))
print(f"COCO: {len(picks)} images copied")

# -------------------------------------------------------------- ExDark
ex = zipfile.ZipFile(DL / "ExDark.zip")
by_cls = {}
for n in ex.namelist():
    parts = n.split("/")
    if parts[0] == "ExDark" and len(parts) == 3 and Path(n).suffix.lower() in IMG_EXT:
        by_cls.setdefault(parts[1], []).append(n)

n_ex = 0
for n in random.sample(by_cls["People"], N_EX_PERSON):
    (OUT / "person" / f"exdark_{Path(n).name}").write_bytes(ex.read(n))
    n_ex += 1
for cls in ("Cat", "Dog", "Chair", "Table", "Bottle"):
    for n in random.sample(by_cls[cls], N_EX_NEG_EACH):
        (OUT / "no_person" / f"exdark_{cls}_{Path(n).name}").write_bytes(ex.read(n))
        n_ex += 1
print(f"ExDark: {n_ex} images copied")
print({s: len(list((OUT / s).iterdir())) for s in ("person", "no_person")})
