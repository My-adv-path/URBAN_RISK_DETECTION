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

counted_ids = set()

vehicle_counts = {
    "car": 0,
    "bus": 0,
    "truck": 0,
    "motorcycle": 0,
    "bicycle": 0
}

previous_positions = {}

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

# Get video dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Put counting line at 60% of frame height
LINE_Y = int(frame_height * 0.60)

print(f"Video size: {frame_width} x {frame_height}")
print(f"Counting line Y: {LINE_Y}")

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

    # Draw counting line
    cv2.line(
        frame,
        (0, LINE_Y),
        (frame_width, LINE_Y),
        (0, 255, 0),
        3
    )

    # Add label for line
    cv2.putText(
        frame,
        "COUNTING LINE",
        (20, LINE_Y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    current_tracked = 0

    if result.boxes.id is not None:

        track_ids = result.boxes.id.int().cpu().tolist()
        class_ids = result.boxes.cls.int().cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        boxes = result.boxes.xyxy.cpu().tolist()

        current_tracked = len(track_ids)

        for track_id, class_id, confidence, box in zip(
            track_ids,
            class_ids,
            confidences,
            boxes
        ):

            class_name = result.names[class_id]

            if class_name not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = box

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Draw center point
            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

            # Draw object label
            cv2.putText(
                frame,
                f"{class_name} ID:{track_id}",
                (int(x1), max(20, int(y1) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

            # Check crossing
            if track_id in previous_positions:

                previous_y = previous_positions[track_id]

                crossed_down = (
                    previous_y < LINE_Y
                    and center_y >= LINE_Y
                )

                crossed_up = (
                    previous_y > LINE_Y
                    and center_y <= LINE_Y
                )

                if (
                    (crossed_down or crossed_up)
                    and track_id not in counted_ids
                ):

                    counted_ids.add(track_id)
                    vehicle_counts[class_name] += 1

                    direction = (
                        "DOWN" if crossed_down else "UP"
                    )

                    print(
                        f"COUNTED: "
                        f"{class_name} "
                        f"ID={track_id} "
                        f"Direction={direction}"
                    )

            previous_positions[track_id] = center_y

    # Display current tracked objects
    cv2.putText(
        frame,
        f"Tracked vehicles: {current_tracked}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Display counts
    y = 70

    for vehicle_type, count in vehicle_counts.items():

        cv2.putText(
            frame,
            f"{vehicle_type}: {count}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        y += 30

    cv2.imshow("ML-2 Vehicle Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\n============================")
print("FINAL VEHICLE COUNTS")
print("============================")

for vehicle_type, count in vehicle_counts.items():
    print(f"{vehicle_type}: {count}")