
import cv2 as cv
from ultralytics import YOLO
import json
from datetime import datetime
import math

INPUT_VIDEO = "ml_person_3/data/traffic.mp4"
OUTPUT_VIDEO = "ml_person_3/data/rash_driving_stable.mp4"
OUTPUT_JSON = "ml_person_3/data/rash_driving_stable_events.json"

SPEED_LIMIT = 45
METERS_PER_PIXEL = 0.05

model = YOLO("yolo11n.pt")

video = cv.VideoCapture(INPUT_VIDEO)

if not video.isOpened():
    print("Error: Could not open video")
    exit()

fps = video.get(cv.CAP_PROP_FPS)
width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 30

writer = cv.VideoWriter(
    OUTPUT_VIDEO,
    cv.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

previous_positions = {}
smoothed_boxes = {}
speed_history = {}
reported_vehicles = set()
events = []

frame_number = 0

while True:
    success, frame = video.read()

    if not success:
        break

    frame_number += 1

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[1, 2, 3, 5, 7],
        conf=0.35,
        verbose=False
    )

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)

            # Smooth the bounding box
            if track_id in smoothed_boxes:
                old_box = smoothed_boxes[track_id]

                alpha = 0.25

                x1 = int(alpha * x1 + (1 - alpha) * old_box[0])
                y1 = int(alpha * y1 + (1 - alpha) * old_box[1])
                x2 = int(alpha * x2 + (1 - alpha) * old_box[2])
                y2 = int(alpha * y2 + (1 - alpha) * old_box[3])

            smoothed_boxes[track_id] = (x1, y1, x2, y2)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            speed = 0

            if track_id in previous_positions:
                old_x, old_y, old_frame = previous_positions[track_id]

                pixel_distance = math.sqrt(
                    (center_x - old_x) ** 2 +
                    (center_y - old_y) ** 2
                )

                time_seconds = (frame_number - old_frame) / fps

                if time_seconds > 0:
                    speed_mps = (
                        pixel_distance * METERS_PER_PIXEL
                    ) / time_seconds

                    current_speed = speed_mps * 3.6

                    if track_id not in speed_history:
                        speed_history[track_id] = []

                    speed_history[track_id].append(current_speed)

                    # Keep only the latest 10 speed values
                    speed_history[track_id] = speed_history[track_id][-10:]

                    speed = sum(speed_history[track_id]) / len(
                        speed_history[track_id]
                    )

            previous_positions[track_id] = (
                center_x,
                center_y,
                frame_number
            )

            rash_driving = speed > SPEED_LIMIT

            if rash_driving and track_id not in reported_vehicles:
                event = {
                    "event_type": "rash_driving",
                    "vehicle_id": track_id,
                    "estimated_speed": round(speed, 2),
                    "speed_limit": SPEED_LIMIT,
                    "timestamp": datetime.now().isoformat(),
                    "confidence": 0.80,
                    "severity": "HIGH"
                }

                events.append(event)
                reported_vehicles.add(track_id)

            color = (0, 0, 255) if rash_driving else (0, 255, 0)

            label = f"ID {track_id} | {speed:.1f} km/h"

            if rash_driving:
                label += " | RASH DRIVING"

            cv.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv.putText(
                frame,
                label,
                (x1, max(y1 - 10, 25)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    writer.write(frame)

video.release()
writer.release()

with open(OUTPUT_JSON, "w") as file:
    json.dump(events, file, indent=2)

print("Stable rash-driving detection completed.")
print("Output video:", OUTPUT_VIDEO)
print("Events JSON:", OUTPUT_JSON)
print("Total unique vehicles flagged:", len(events))