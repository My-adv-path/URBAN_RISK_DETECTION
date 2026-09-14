import cv2
from ultralytics import YOLO

VIDEO_PATH = "Videos/traffic.mp4"

model = YOLO("yolo26n.pt")

VEHICLE_CLASSES = {
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle"
}

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video size: {frame_width} x {frame_height}")

# -----------------------------------
# ROAD REGION
# -----------------------------------

# For your 1920 x 1080 video.
# We will start with a simple rectangular ROI.
ROI_X1 = 200
ROI_Y1 = 300
ROI_X2 = 1720
ROI_Y2 = 950

# Maximum number of vehicles that we
# consider "100% occupied" for prototype.
MAX_VEHICLES = 30

while True:

    success, frame = cap.read()

    if not success:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.35,
        verbose=False
    )

    result = results[0]

    vehicle_centers = []

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

            if class_name not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = box

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Check whether center is inside ROI
            inside_roi = (
                ROI_X1 <= center_x <= ROI_X2
                and ROI_Y1 <= center_y <= ROI_Y2
            )

            if inside_roi:

                vehicle_centers.append(
                    (center_x, center_y)
                )

                # Draw green center point
                cv2.circle(
                    frame,
                    (center_x, center_y),
                    6,
                    (0, 255, 0),
                    -1
                )

            else:

                # Draw normal center point
                cv2.circle(
                    frame,
                    (center_x, center_y),
                    4,
                    (0, 0, 255),
                    -1
                )

    # -----------------------------------
    # DENSITY
    # -----------------------------------

    active_vehicles = len(vehicle_centers)

    density = active_vehicles / MAX_VEHICLES

    density = min(density, 1.0)

    # Convert to percentage
    density_percent = density * 100

    if density < 0.30:
        density_level = "LOW"

    elif density < 0.60:
        density_level = "MEDIUM"

    elif density < 0.80:
        density_level = "HIGH"

    else:
        density_level = "VERY HIGH"

    # -----------------------------------
    # DRAW ROI
    # -----------------------------------

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

    # -----------------------------------
    # DISPLAY DENSITY
    # -----------------------------------

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