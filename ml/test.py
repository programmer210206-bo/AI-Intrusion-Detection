
from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

THRESHOLD = 0.70

results = model("test.jpg")

person_detected = False

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = result.names[class_id]

        if class_name == "person" and confidence >= THRESHOLD:
            person_detected = True
            print(f"Person detected! Confidence: {confidence:.2f}")

    annotated_image = result.plot(conf=True)
    cv2.imshow("AI Intrusion Detection - YOLO", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if person_detected:
    print("HUMAN INTRUSION DETECTED")
else:
    print("NO HUMAN DETECTED")