# ML part: human verification

Pipeline: image -> YOLO11m (person only) -> size/shape filters -> threshold -> `person: true/false`

| File | Purpose |
|---|---|
| `detector.py` | `verify_person(image)` - the core function |
| `evaluate.py` | accuracy report, threshold sweep, mistakes list (`data/person`, `data/no_person`) |
| `webcam.py` | live demo with 3-of-5-frame confirmation |
| `server.py` | `POST /verify` for the ESP32-CAM (raw JPEG or multipart `file`) |
| `client_test.py` | pretends to be the ESP32-CAM |

## Run
```
venv\Scripts\activate
python detector.py test.jpg --show
python evaluate.py
python webcam.py
uvicorn server:app --host 0.0.0.0 --port 8000
python client_test.py data\person\zidane.jpg
```

## Tuning (environment variables)
`MODEL_NAME` (yolo11s/m/l/x.pt, rtdetr-l.pt), `CONF_THRESHOLD`, `IMG_SIZE`,
`MIN_BOX_AREA`, `MAX_ASPECT`, `USE_TTA=1`, `ZOOM_OUT_PASS`, `ENHANCE_DARK`.
Example: `$env:MODEL_NAME="yolo11x.pt"; python evaluate.py`

## Next
Add real images (ESP32-CAM frames, pets, empty room, night) to `data/` and re-run `evaluate.py`.
