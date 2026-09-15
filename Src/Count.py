import cv2
from ultralytics import YOLO

VIDEO_PATH = "Videos/indian_traffic.mp4"
MODEL_PATH = "Models/UVH-26-MV-YOLOv11-S.pt"

model = YOLO(MODEL_PATH)

CLASS_MAP = {
    "Hatchback": "car",
    "Sedan": "car",
    "SUV": "car",
    "MUV": "car",
    "Van": "car",
    "Three-wheeler": "auto",
    "Bus": "bus",
    "Mini-bus": "bus",
    "Truck": "truck",
    "LCV": "truck",
    "tempo-traveller": "truck",
    "Two-wheeler": "motorcycle"
}

counts = {
    "car": 0,
    "auto": 0,
    "bus": 0,
    "truck": 0,
    "motorcycle": 0
}

counted = set()
previous_x = {}

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    raise SystemExit

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

line_x = int(width * 0.20)

print(f"Video size: {width} x {height}")
print(f"Counting line X: {line_x}")

while True:
    success, frame = cap.read()

    if not success:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.40,
        verbose=False
    )

    result = results[0]
    tracked = 0

    cv2.line(
        frame,
        (line_x, 0),
        (line_x, height),
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        "COUNTING LINE",
        (line_x + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    if result.boxes is not None and result.boxes.id is not None:
        ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()
        boxes = result.boxes.xyxy.cpu().tolist()

        for track_id, class_id, box in zip(ids, classes, boxes):
            raw_name = result.names[class_id]
            name = CLASS_MAP.get(raw_name)

            if name is None:
                continue

            tracked += 1

            x1, y1, x2, y2 = box

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"{name} {track_id}",
                (int(x1), max(20, int(y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

            cv2.circle(
                frame,
                (cx, cy),
                4,
                (0, 0, 255),
                -1
            )

            if track_id in previous_x:
                old_x = previous_x[track_id]

                crossed = (
                    (old_x < line_x <= cx)
                    or
                    (old_x > line_x >= cx)
                )

                if crossed and track_id not in counted:
                    counted.add(track_id)
                    counts[name] += 1

                    direction = "RIGHT" if cx > old_x else "LEFT"

                    print(
                        f"COUNTED: {name} "
                        f"ID={track_id} "
                        f"Direction={direction}"
                    )

            previous_x[track_id] = cx

    cv2.putText(
        frame,
        f"Tracked vehicles: {tracked}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    y = 70

    for name, value in counts.items():
        cv2.putText(
            frame,
            f"{name}: {value}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        y += 30

    cv2.imshow(
        "ML-2 Vehicle Counting",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\n============================")
print("FINAL VEHICLE COUNTS")
print("============================")

for name, value in counts.items():
    print(f"{name}: {value}")