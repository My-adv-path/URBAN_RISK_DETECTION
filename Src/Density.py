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

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    raise SystemExit

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video size: {frame_width} x {frame_height}")

ROI_X1 = 100
ROI_Y1 = 180
ROI_X2 = 1200
ROI_Y2 = 620

MAX_VEHICLES = 50

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
    vehicle_centers = []

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
            vehicle_type = CLASS_MAP.get(raw_class)

            if vehicle_type is None:
                continue

            x1, y1, x2, y2 = box

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            inside_roi = (
                ROI_X1 <= center_x <= ROI_X2
                and ROI_Y1 <= center_y <= ROI_Y2
            )

            if inside_roi:
                vehicle_centers.append(
                    (center_x, center_y)
                )

                cv2.circle(
                    frame,
                    (center_x, center_y),
                    6,
                    (0, 255, 0),
                    -1
                )

                cv2.putText(
                    frame,
                    vehicle_type,
                    (center_x + 8, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.circle(
                    frame,
                    (center_x, center_y),
                    4,
                    (0, 0, 255),
                    -1
                )

    active_vehicles = len(vehicle_centers)

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

    cv2.rectangle(
        frame,
        (ROI_X1, ROI_Y1),
        (ROI_X2, ROI_Y2),
        (255, 0, 0),
        3
    )

    cv2.putText(
        frame,
        "ROAD ROI",
        (ROI_X1, ROI_Y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        f"Vehicles: {active_vehicles}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Density: {density_percent:.1f}%",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Level: {density_level}",
        (30, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "ML-2 Traffic Density",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\nDensity analysis complete.")