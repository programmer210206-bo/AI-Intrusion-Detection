"""Collect training/test photos straight from the ESP32-CAM.

    python collect_camera.py person 30        # 30 photos with a person   -> data/person/
    python collect_camera.py no_person 30     # 30 photos without anyone  -> data/no_person/
    python collect_camera.py screen 30        # 30 photos of people on screens/prints -> data/screen_negatives/

A photo is taken every 3 seconds, so move, pose and change the lighting between shots.
Close every camera preview page first (the camera serves one request at a time).
"""
import os
import sys
import time
from pathlib import Path

import requests

FOLDERS = {"person": "data/person", "no_person": "data/no_person", "screen": "data/screen_negatives"}
if len(sys.argv) < 2 or sys.argv[1] not in FOLDERS:
    raise SystemExit(__doc__)
kind, count = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 30

url = os.getenv("CAMERA_URL", "http://172.20.10.3/capture")
raw = input("Paste the camera token and press Enter: ")
token = "".join(c for c in raw if 32 < ord(c) < 127 and c not in "\"'")

out = Path(FOLDERS[kind])
out.mkdir(parents=True, exist_ok=True)
saved = 0
print(f"Saving {count} '{kind}' photos to {out}. Get ready...")
time.sleep(3)
while saved < count:
    try:
        r = requests.post(url, headers={"X-Camera-Token": token}, timeout=20)
        ok = r.status_code == 200 and r.content.rstrip(b"\x00").endswith(b"\xff\xd9")
    except requests.RequestException:
        ok = False
    if ok:
        saved += 1
        (out / f"esp32_{kind}_{time.strftime('%H%M%S')}_{saved:03d}.jpg").write_bytes(r.content)
        print(f"  {saved}/{count} saved")
    else:
        print("  photo failed, retrying...")
    time.sleep(3)
print("Done. Next: python evaluate.py")
