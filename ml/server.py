"""HTTP server the ESP32 (or any client) calls to verify a human.

Run:   uvicorn server:app --host 0.0.0.0 --port 8000
ESP32: POST raw JPEG bytes (Content-Type: image/jpeg) to http://<laptop-ip>:8000/verify
       or multipart upload with field name "file".
Reply: {"person": true/false, "confidence": 0.91, "count": 1, "boxes": [...], "ms": 120.5}
Every request is saved to captures/ (annotated) for debugging and the demo.
"""
import os
import time
from pathlib import Path

import cv2
import requests
from fastapi import FastAPI, HTTPException, Request

import detector

app = FastAPI(title="Intrusion Detection - Human Verification")
CAPTURES = Path("captures")
CAPTURES.mkdir(exist_ok=True)

# ESP32-CAM photo URL (the Arduino CameraWebServer example serves a single JPEG at /capture)
CAMERA_URL = os.getenv("CAMERA_URL", "http://172.20.10.3/capture")
CAMERA_TIMEOUT = float(os.getenv("CAMERA_TIMEOUT", "20"))  # the ESP32-CAM can take several seconds
CAMERA_RETRIES = int(os.getenv("CAMERA_RETRIES", "3"))
CHECK_FRAMES = int(os.getenv("CHECK_FRAMES", "2"))
# Our camera firmware: POST /capture with an X-Camera-Token header. Set CAMERA_TOKEN in your own
# terminal (never in the code). For a Basic-login camera use CAMERA_USER / CAMERA_PASS instead.
CAMERA_METHOD = os.getenv("CAMERA_METHOD", "POST")
_token = "".join(c for c in os.getenv("CAMERA_TOKEN", "") if 32 < ord(c) < 127 and c not in "\"'")
CAMERA_HEADERS = {"X-Camera-Token": _token} if _token else {}
CAMERA_AUTH = (os.environ["CAMERA_USER"], os.getenv("CAMERA_PASS", "")) if os.getenv("CAMERA_USER") else None
LAST: dict = {}


@app.on_event("startup")
def _load():
    detector.get_model()


@app.get("/health")
def health():
    return {"status": "ok", "model": detector.MODEL_NAME, "threshold": detector.CONF_THRESHOLD}


@app.post("/verify")
async def verify(request: Request):
    ctype = request.headers.get("content-type", "")
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if upload is None:
            raise HTTPException(400, "multipart field 'file' missing")
        data = await upload.read()
    else:
        data = await request.body()
    if not data:
        raise HTTPException(400, "empty body")

    try:
        img = detector.load_image(data)
    except ValueError:
        raise HTTPException(400, "not a valid image")  # truncated ESP32 frames happen

    return _verify_and_save(img)


def _verify_and_save(img):
    result = detector.verify_person(img)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = "PERSON" if result["person"] else "none"
    cv2.imwrite(str(CAPTURES / f"{stamp}_{tag}_{result['confidence']:.2f}.jpg"), detector.annotate(img, result))
    result["time"] = stamp
    return result


@app.api_route("/check", methods=["GET", "POST"])
def check():
    """Sensor ESP32 calls this when PIR + ultrasonic agree. The PC fetches a photo from the
    ESP32-CAM, runs the model and returns the verdict (the ESP32 then sounds the buzzer).
    Takes CHECK_FRAMES photos; any frame with a person confirms the intrusion."""
    results = []
    for _ in range(CHECK_FRAMES):
        img, err = None, None
        for _attempt in range(CAMERA_RETRIES):  # the ESP32-CAM occasionally stalls; try again
            try:
                r = requests.request(CAMERA_METHOD, CAMERA_URL, timeout=CAMERA_TIMEOUT,
                                     headers=CAMERA_HEADERS, auth=CAMERA_AUTH)
                if r.status_code in (401, 403):  # a wrong token will not fix itself
                    raise HTTPException(502, "camera rejected the token (401): check CAMERA_TOKEN")
                r.raise_for_status()
                if not r.content.rstrip(b"\x00").endswith(b"\xff\xd9"):
                    raise ValueError("photo was cut off (JPEG has no end marker)")
                img = detector.load_image(r.content)
                break
            except (requests.RequestException, ValueError) as e:
                err = e
                time.sleep(1.5)  # let the camera finish whatever it was sending
        if img is None:
            raise HTTPException(502, f"could not get a photo from the camera ({CAMERA_URL}): {err}")
        results.append(_verify_and_save(img))
    best = max(results, key=lambda x: x["confidence"])
    best["person"] = any(x["person"] for x in results)
    best["frames"] = len(results)
    LAST.update(best)
    if best["person"]:
        print(f"\a*** ALERT: HUMAN INTRUSION CONFIRMED ({best['confidence']:.2f}) at {best['time']} ***", flush=True)
    else:
        print(f"[{best['time']}] sensors triggered, no person in image - ignored", flush=True)
    return best


@app.get("/status")
def status():
    """Last /check verdict, for a dashboard or for polling."""
    return LAST or {"status": "no checks yet"}
