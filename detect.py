import cv2
from ultralytics import YOLO
from collections import deque

model = YOLO("runs/detect/train/weights/best.pt")

THREAT_LEVEL = {
    "drones":     ("HIGH THREAT",   (0, 0, 255)),
    "plane":      ("MEDIUM THREAT", (0, 165, 255)),
    "helicopter": ("MEDIUM THREAT", (0, 165, 255)),
    "birds":      ("LOW THREAT",    (0, 255, 0)),
    "0":          ("UNKNOWN",       (128, 128, 128)),
}

history = deque(maxlen=5)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.45, verbose=False)[0]

    for box in results.boxes:
        cls_name = model.names[int(box.cls)]
        conf = float(box.conf)
        threat, color = THREAT_LEVEL.get(cls_name, ("UNKNOWN", (128,128,128)))
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        label = f"{cls_name} {conf:.0%}"
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    current = [model.names[int(b.cls)] for b in results.boxes]
    history.append(current)

    all_recent = [cls for frame_dets in history for cls in frame_dets]

    if all_recent.count("drones") >= 2:
        summary, scol = "!! HIGH THREAT — DRONE DETECTED !!", (0, 0, 255)
    elif any(t in all_recent for t in ["plane", "helicopter"]):
        summary, scol = "MEDIUM THREAT — AIRCRAFT DETECTED", (0, 165, 255)
    elif all_recent:
        summary, scol = "LOW THREAT — BIRDS DETECTED", (0, 255, 0)
    else:
        summary, scol = "AIRSPACE CLEAR", (200, 200, 200)

    cv2.rectangle(frame, (0,0), (frame.shape[1], 55), (20,20,20), -1)
    cv2.putText(frame, summary, (15, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, scol, 2)

    cv2.imshow("Aerial Threat Detection System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()