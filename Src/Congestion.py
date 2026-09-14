import cv2
import math
from collections import deque
from ultralytics import YOLO

# ===================================
# SETTINGS
# ===================================

VIDEO_PATH = "Videos/traffic.mp4"

# Load YOLO model
model = YOLO("yolo26n.pt")

# Vehicle classes we currently consider
VEHICLE_CLASSES = {
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle"
}

# ===================================
# OPEN VIDEO
# ===================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

print(f"Video size: {frame_width} x {frame_height}")
print(f"FPS: {fps:.2f}")

# ===================================
# CREATE RESIZABLE VIDEO WINDOW
# ===================================

cv2.namedWindow(
    "ML-2 Traffic Congestion",
    cv2.WINDOW_NORMAL
)

# Resize the display window so the complete
# video fits on the screen.
cv2.resizeWindow(
    "ML-2 Traffic Congestion",
    1000,
    600
)

# ===================================
# ROAD REGION
# ===================================

ROI_X1 = 200
ROI_Y1 = 300
ROI_X2 = 1720
ROI_Y2 = 950

# Prototype reference capacity
MAX_VEHICLES = 30

# ===================================
# MOVEMENT TRACKING
# ===================================

previous_positions = {}

# Store recent movement values
movement_history = deque(maxlen=30)

# Number of frames for which congestion
# conditions must persist
congestion_frames = 0

CONGESTION_PERSISTENCE = int(fps * 3)

# ===================================
# MAIN VIDEO LOOP
# ===================================

while True:

    success, frame = cap.read()

    if not success:
        break

    # ===================================
    # YOLO + BYTE TRACK
    # ===================================

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.35,
        verbose=False
    )

    result = results[0]

    active_vehicles = 0
    movements = []

    # ===================================
    # PROCESS TRACKED OBJECTS
    # ===================================

    if result.boxes.id is not None:

        track_ids = result.boxes.id.int().cpu().tolist()
        class_ids = result.boxes.cls.int().cpu().tolist()
        boxes = result.boxes.xyxy.cpu().tolist()

        for track_id, class_id, box in zip(
            track_ids,
            class_ids,
            boxes
        ):

            class_name = result.names[class_id]

            # Ignore pedestrians for traffic analytics
            if class_name not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = box

            # Calculate center of bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # ===================================
            # CHECK WHETHER VEHICLE IS INSIDE ROI
            # ===================================

            inside_roi = (
                ROI_X1 <= center_x <= ROI_X2
                and ROI_Y1 <= center_y <= ROI_Y2
            )

            if not inside_roi:
                continue

            active_vehicles += 1

            # ===================================
            # MOVEMENT ESTIMATION
            # ===================================

            if track_id in previous_positions:

                previous_x, previous_y = previous_positions[track_id]

                movement = math.sqrt(
                    (center_x - previous_x) ** 2
                    +
                    (center_y - previous_y) ** 2
                )

                movements.append(movement)

            # Save current position
            previous_positions[track_id] = (
                center_x,
                center_y
            )

            # ===================================
            # DRAW VEHICLE CENTER
            # ===================================

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 255, 0),
                -1
            )

            # ===================================
            # DRAW TRACK ID
            # ===================================

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

    # ===================================
    # DENSITY CALCULATION
    # ===================================

    density = active_vehicles / MAX_VEHICLES

    # Prevent density from exceeding 100%
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

    # ===================================
    # AVERAGE VEHICLE MOVEMENT
    # ===================================

    if movements:

        average_movement = (
            sum(movements)
            / len(movements)
        )

    else:

        average_movement = 0

    movement_history.append(
        average_movement
    )

    if movement_history:

        smoothed_movement = (
            sum(movement_history)
            / len(movement_history)
        )

    else:

        smoothed_movement = 0

    # ===================================
    # CONGESTION LOGIC
    # ===================================

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

    else:

        if density >= 0.60:

            congestion_level = "HEAVY TRAFFIC"

        else:

            congestion_level = "NORMAL"

    # ===================================
    # DRAW ROAD ROI
    # ===================================

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

    # ===================================
    # DISPLAY ANALYTICS
    # ===================================

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

    # ===================================
    # SHOW VIDEO
    # ===================================

    cv2.imshow(
        "ML-2 Traffic Congestion",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ===================================
# CLEANUP
# ===================================

cap.release()
cv2.destroyAllWindows()

print("\nCongestion analysis complete.")