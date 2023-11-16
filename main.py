import cv2
import cvzone
import math
from ultralytics import YOLO

# Load YOLOv5 model
model = YOLO("best.pt")  # Replace with the path to your YOLOv5 model for "fire" and "smoke"

# Class names for detection
class_names = ['fire', 'smoke']

# Color settings for bounding boxes
colors = [(0, 0, 255), (0, 255, 0)]  # Red for "fire," Green for "smoke"

# Set up the video capture from a file
cap = cv2.VideoCapture("samplev3.mp4")  # Replace with the path to your video file

while True:
    success, img = cap.read()

    if not success:
        break

    # Detect objects using the YOLOv5 model
    results = model(img, stream=True)

    # ...

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            current_class = class_names[cls]

            print(f"Detected {current_class} with confidence {conf} at coordinates ({x1}, {y1}, {x2}, {y2})")

            if conf > 0.05:
                color = colors[cls]
                cvzone.putTextRect(img, f'{current_class} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=color,
                                   colorT=(255, 255, 255), colorR=color, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

    # ...

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()

