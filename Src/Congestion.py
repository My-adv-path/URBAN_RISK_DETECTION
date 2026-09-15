import cv2
import math
from collections import deque
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

VEHICLE_CLASSES = {
    "car",
    "auto",
    "bus",
    "truck",
    "motorcycle"
}

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    raise SystemExit

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

print(f"Video size: {frame_width} x {frame_height}")
print(f"FPS: {fps:.2f}")

cv2.namedWindow(
    "ML-2 Traffic Congestion",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "ML-2 Traffic Congestion",
    1000,
    600
)

ROI_X1 = 100
ROI_Y1 = 180
ROI_X2 = 1200
ROI_Y2 = 620

MAX_VEHICLES = 30

previous_positions = {}

movement_history = deque(maxlen=30)

congestion_frames = 0

CONGESTION_PERSISTENCE = int(fps * 3)

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

    active_vehicles = 0
    movements = []

    if result.boxes is not None and result.boxes.id is not None:
        track_ids = result.boxes.id.int().cpu().tolist()
        class_ids = result.boxes.cls.int().cpu().tolist()
        boxes = result.boxes.xyxy.cpu().tolist()

        for track_id, class_id, box in zip(
            track_ids,
            class_ids,
            boxes
        ):
            raw_class = result.names[int(class_id)]
            class_name = CLASS_MAP.get(raw_class)

            if class_name not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = box

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            inside_roi = (
                ROI_X1 <= center_x <= ROI_X2
                and ROI_Y1 <= center_y <= ROI_Y2
            )

            if not inside_roi:
                continue

            active_vehicles += 1

            if track_id in previous_positions:
                previous_x, previous_y = previous_positions[track_id]

                movement = math.sqrt(
                    (center_x - previous_x) ** 2
                    + (center_y - previous_y) ** 2
                )

                movements.append(movement)

            previous_positions[track_id] = (
                center_x,
                center_y
            )

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 255, 0),
                -1
            )

            cv2.putText(
                frame,
                f"{class_name} ID:{track_id}",
                (
                    int(x1),
                    max(20, int(y1) - 10)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

    density = active_vehicles / MAX_VEHICLES
    density = min(density, 1.0)

    density_percent = density * 100

    if density < 0.30:
        density_level = "LOW"
    elif density < 0.60:
        density_level = "MEDIUM"
    elif density < 0.80:
        density_level = "HIGH"
    else:
        density_level = "VERY HIGH"

    if movements:
        average_movement = (
            sum(movements) / len(movements)
        )
    else:
        average_movement = 0

    movement_history.append(
        average_movement
    )

    smoothed_movement = (
        sum(movement_history) /
        len(movement_history)
    )

    HIGH_DENSITY_THRESHOLD = 0.60
    LOW_MOVEMENT_THRESHOLD = 3.0

    congestion_condition = (
        density >= HIGH_DENSITY_THRESHOLD
        and smoothed_movement <= LOW_MOVEMENT_THRESHOLD
    )

    if congestion_condition:
        congestion_frames += 1
    else:
        congestion_frames = max(
            0,
            congestion_frames - 2
        )

    if congestion_frames >= CONGESTION_PERSISTENCE:
        congestion_level = "HIGH CONGESTION"
    elif congestion_frames >= CONGESTION_PERSISTENCE // 2:
        congestion_level = "POSSIBLE CONGESTION"
    elif density >= 0.60:
        congestion_level = "HEAVY TRAFFIC"
    else:
        congestion_level = "NORMAL"

    cv2.rectangle(
        frame,
        (ROI_X1, ROI_Y1),
        (ROI_X2, ROI_Y2),
        (255, 0, 0),
        3
    )

    cv2.putText(
        frame,
        "TRAFFIC ROI",
        (ROI_X1, ROI_Y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        f"Vehicles: {active_vehicles}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Density: {density_percent:.1f}%",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Density Level: {density_level}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Movement: {smoothed_movement:.2f}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Traffic Status: {congestion_level}",
        (20, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "ML-2 Traffic Congestion",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\nCongestion analysis complete.")