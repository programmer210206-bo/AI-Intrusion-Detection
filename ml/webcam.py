"""Live human detection from the laptop webcam. Press q to quit.

An alert needs a person in at least CONFIRM_FRAMES of the last WINDOW frames,
which removes single-frame flicker (a major source of false alarms).
"""
import sys
from collections import deque

import cv2

import detector

WINDOW, CONFIRM_FRAMES = 5, 3
cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit(f"Cannot open camera {cam_index}")

history = deque(maxlen=WINDOW)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    r = detector.verify_person(frame)
    history.append(r["person"])
    confirmed = sum(history) >= CONFIRM_FRAMES

    view = detector.annotate(frame, r)
    if confirmed:
        # orange frame + banner, distinct from the red person boxes
        cv2.rectangle(view, (0, 0), (view.shape[1] - 1, view.shape[0] - 1), (0, 140, 255), 3)
        cv2.rectangle(view, (0, view.shape[0] - 70), (view.shape[1], view.shape[0] - 35), (0, 140, 255), -1)
        cv2.putText(view, "ALERT: HUMAN CONFIRMED", (10, view.shape[0] - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(view, f"{r['ms']:.0f} ms", (10, view.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("AI Intrusion Detection (q to quit)", view)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
