"""Simulates the ESP32-CAM: posts a JPEG to the server and prints the verdict.

    python client_test.py test.jpg [http://localhost:8000]
"""
import sys

import requests

path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
base = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"

with open(path, "rb") as f:
    r = requests.post(f"{base}/verify", data=f.read(), headers={"Content-Type": "image/jpeg"}, timeout=10)
print(r.status_code, r.json())
