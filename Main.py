import cv2
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict, deque
from ultralytics import YOLO

MODEL_PATH = "Models/UVH-26-MV-YOLOv11-S.pt"
VIDEO_PATH = "Videos/indian_traffic.mp4"
OUTPUT_PATH = "Outputs/final_traffic_analysis.mp4"
EVENTS_PATH = "Outputs/events.json"
EVENT_IMAGE_DIR = "Outputs/events"
TRACKER_PATH = "Configs/botsort_custom.yaml"

CAMERA_LATITUDE = 13.08
CAMERA_LONGITUDE = 80.27

ROI_X1 = 100
ROI_Y1 = 180
ROI_X2 = 1200
ROI_Y2 = 620

COUNTING_LINE_RATIO = 0.18
COUNTING_ZONE_WIDTH = 120

MAX_VEHICLES = 30
MIN_TRACK_FRAMES = 5

CONGESTION_STABLE_SECONDS = 3

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

events = []

counted_ids = set()
previous_x = {}
track_frames = defaultdict(int)
track_history = defaultdict(lambda: deque(maxlen=10))
movement_history = deque(maxlen=30)

previous_congestion_state = "NORMAL"
candidate_congestion_state = "NORMAL"
candidate_state_frames = 0

camera_start_time = datetime.now()

os.makedirs("Outputs", exist_ok=True)
os.makedirs(EVENT_IMAGE_DIR, exist_ok=True)

for filename in os.listdir(EVENT_IMAGE_DIR):
    file_path = os.path.join(EVENT_IMAGE_DIR, filename)

    if os.path.isfile(file_path):
        os.remove(file_path)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    raise SystemExit

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 25

line_x = int(width * COUNTING_LINE_RATIO)
zone_left = max(0, line_x - COUNTING_ZONE_WIDTH)
zone_right = min(width, line_x + COUNTING_ZONE_WIDTH)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

print(f"Video size: {width} x {height}")
print(f"FPS: {fps:.1f}")
print(f"Counting zone: X={zone_left} to X={zone_right}")
print("Starting integrated traffic analysis...")

frame_number = 0
event_number = 0

while True:
    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    results = model.track(
        frame,
        persist=True,
        tracker=TRACKER_PATH,
        conf=0.40,
        iou=0.55,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    active_vehicles = 0
    moving_vehicles = 0
    roi_confidences = []

    if result.boxes is not None and result.boxes.id is not None:
        track_ids = result.boxes.id.int().cpu().tolist()
        class_ids = result.boxes.cls.int().cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        boxes = result.boxes.xyxy.cpu().tolist()

        for track_id, class_id, confidence, box in zip(
            track_ids,
            class_ids,
            confidences,
            boxes
        ):
            raw_class = result.names[int(class_id)]
            vehicle_type = CLASS_MAP.get(raw_class)

            if vehicle_type is None:
                continue

            track_frames[track_id] += 1

            x1, y1, x2, y2 = box

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            inside_roi = (
                ROI_X1 <= center_x <= ROI_X2
                and ROI_Y1 <= center_y <= ROI_Y2
            )

            if inside_roi:
                active_vehicles += 1
                roi_confidences.append(float(confidence))

            movement = 0.0

            if len(track_history[track_id]) > 0:
                previous_position = track_history[track_id][-1]

                previous_center_x = previous_position[0]
                previous_center_y = previous_position[1]

                movement = (
                    (center_x - previous_center_x) ** 2
                    + (center_y - previous_center_y) ** 2
                ) ** 0.5

            track_history[track_id].append(
                (center_x, center_y)
            )

            if inside_roi and movement > 2.0:
                moving_vehicles += 1

            if track_id in previous_x:
                old_x = previous_x[track_id]

                crossed_zone = (
                    (
                        old_x < zone_left
                        and center_x >= zone_left
                    )
                    or
                    (
                        old_x > zone_right
                        and center_x <= zone_right
                    )
                )

                stable_track = (
                    track_frames[track_id] >= MIN_TRACK_FRAMES
                )

                if (
                    crossed_zone
                    and stable_track
                    and track_id not in counted_ids
                ):
                    counted_ids.add(track_id)
                    counts[vehicle_type] += 1
                    event_number += 1

                    crop_padding = 20

                    crop_x1 = max(
                        0,
                        int(x1) - crop_padding
                    )

                    crop_y1 = max(
                        0,
                        int(y1) - crop_padding
                    )

                    crop_x2 = min(
                        width,
                        int(x2) + crop_padding
                    )

                    crop_y2 = min(
                        height,
                        int(y2) + crop_padding
                    )

                    event_image_path = (
                        f"{EVENT_IMAGE_DIR}/event_{event_number:04d}.jpg"
                    )

                    if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                        vehicle_crop = frame[
                            crop_y1:crop_y2,
                            crop_x1:crop_x2
                        ]

                        cv2.imwrite(
                            event_image_path,
                            vehicle_crop
                        )
                    else:
                        event_image_path = None

                    timestamp = (
                        camera_start_time
                        + timedelta(seconds=frame_number / fps)
                    ).isoformat()

                    event = {
                        "event_type": vehicle_type,
                        "confidence": round(float(confidence), 3),
                        "bbox": [
                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2)
                        ],
                        "timestamp": timestamp,
                        "image": event_image_path,
                        "latitude": CAMERA_LATITUDE,
                        "longitude": CAMERA_LONGITUDE
                    }

                    events.append(event)

                    direction = (
                        "RIGHT"
                        if center_x > old_x
                        else "LEFT"
                    )

                    print(
                        f"EVENT: {vehicle_type} "
                        f"Confidence={confidence:.2f} "
                        f"Direction={direction}"
                    )

            previous_x[track_id] = center_x

            box_color = (
                (0, 255, 255)
                if inside_roi
                else (255, 255, 255)
            )

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                box_color,
                2
            )

            cv2.putText(
                frame,
                f"{vehicle_type} {confidence:.2f}",
                (int(x1), max(20, int(y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                2
            )

    density = min(
        active_vehicles / MAX_VEHICLES,
        1.0
    )

    density_percent = density * 100

    if density < 0.30:
        density_level = "LOW"
    elif density < 0.60:
        density_level = "MEDIUM"
    elif density < 0.80:
        density_level = "HIGH"
    else:
        density_level = "VERY HIGH"

    movement_history.append(moving_vehicles)

    if movement_history:
        average_movement = (
            sum(movement_history)
            / len(movement_history)
        )
    else:
        average_movement = 0

    if density >= 0.60 and average_movement <= 3.0:
        raw_congestion_state = "HIGH CONGESTION"
    elif density >= 0.60:
        raw_congestion_state = "HEAVY TRAFFIC"
    elif density >= 0.30:
        raw_congestion_state = "MODERATE TRAFFIC"
    else:
        raw_congestion_state = "NORMAL"

    if raw_congestion_state == candidate_congestion_state:
        candidate_state_frames += 1
    else:
        candidate_congestion_state = raw_congestion_state
        candidate_state_frames = 1

    stable_threshold = int(
        fps * CONGESTION_STABLE_SECONDS
    )

    if candidate_state_frames >= stable_threshold:
        if candidate_congestion_state != previous_congestion_state:
            event_image_path = (
                f"{EVENT_IMAGE_DIR}/event_{event_number + 1:04d}.jpg"
            )

            cv2.imwrite(
                event_image_path,
                frame
            )

            event_number += 1

            timestamp = (
                camera_start_time
                + timedelta(seconds=frame_number / fps)
            ).isoformat()

            congestion_confidence = (
                0.5 * density
                + 0.5 * (
                    1.0
                    - min(
                        average_movement / 10.0,
                        1.0
                    )
                )
            )

            event = {
                "event_type": "traffic_congestion",
                "confidence": round(
                    max(
                        0.0,
                        min(
                            congestion_confidence,
                            1.0
                        )
                    ),
                    3
                ),
                "bbox": [],
                "timestamp": timestamp,
                "image": event_image_path,
                "latitude": CAMERA_LATITUDE,
                "longitude": CAMERA_LONGITUDE
            }

            events.append(event)

            print(
                f"CONGESTION EVENT: "
                f"{candidate_congestion_state}"
            )

            previous_congestion_state = (
                candidate_congestion_state
            )

    cv2.putText(
        frame,
        f"Vehicles: {active_vehicles}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Density: {density_percent:.1f}% ({density_level})",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Moving: {moving_vehicles}",
        (30, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Congestion: {previous_congestion_state}",
        (30, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    y = 190

    for vehicle_type, value in counts.items():
        cv2.putText(
            frame,
            f"{vehicle_type}: {value}",
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        y += 30

    writer.write(frame)

    cv2.imshow(
        "ML-2 Integrated Traffic Analysis",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

with open(EVENTS_PATH, "w") as file:
    json.dump(events, file, indent=4)

print("\n============================")
print("FINAL TRAFFIC ANALYSIS")
print("============================")

for vehicle_type, value in counts.items():
    print(f"{vehicle_type}: {value}")

print(f"Final congestion state: {previous_congestion_state}")
print(f"Output video: {OUTPUT_PATH}")
print(f"Events JSON: {EVENTS_PATH}")
print(f"Total events: {len(events)}")
print("Integrated traffic analysis complete!")