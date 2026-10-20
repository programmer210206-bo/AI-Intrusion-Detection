"""Test the ESP32-CAM directly (no model, no server).

    python test_camera.py

It asks for the camera token, takes one photo and saves it as test_cam.jpg.
"""
import os

import requests

url = os.getenv("CAMERA_URL", "http://172.20.10.3/capture")
print(f"Camera address: {url}")
raw = input("Paste the camera token and press Enter: ")
# drop hidden/non-ASCII characters (BOM, zero-width, curly quotes) that sneak in when pasting
token = "".join(c for c in raw if 32 < ord(c) < 127 and c not in "\"'")
if token != raw.strip():
    print("(removed hidden or quote characters from what you pasted)")

try:
    r = requests.post(url, headers={"X-Camera-Token": token}, timeout=20)
except requests.RequestException as e:
    print("\nRESULT: could not reach the camera.")
    print("-> Is the camera powered and on the same hotspot? Did its IP address change?")
    print("   (The Arduino Serial Monitor prints 'Camera URL: http://...')")
    print("Details:", e)
    raise SystemExit(1)

print("\nHTTP status:", r.status_code)
if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/jpeg"):
    open("test_cam.jpg", "wb").write(r.content)
    print(f"RESULT: SUCCESS - photo saved as test_cam.jpg ({len(r.content)} bytes). Open it to check.")
elif r.status_code == 401:
    print("RESULT: the camera rejected the token (401).")
    print("-> The token you typed differs from the one flashed on the board.")
    print("   Compare it with CAMERA_TOKEN in your Arduino sketch, then re-upload the sketch to be sure.")
else:
    print("RESULT: unexpected reply:", r.text[:200])
