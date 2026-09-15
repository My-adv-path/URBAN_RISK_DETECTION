import cv2 as cv
import json
from ultralytics import YOLO

# Load the vehicle detection model
vehicle_model = YOLO("yolo11n.pt")

# Open the video
video = cv.VideoCapture("ml_person_3/data/anpr_test.mp4")

if not video.isOpened():
    print("Video cannot be accessed")
    exit()

# Video information
width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv.CAP_PROP_FPS)

# Output video
output = cv.VideoWriter(
    "ml_person_3/data/incident_output.mp4",
    cv.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# Virtual incident-detection line
LINE_POSITION = 0.50
line_y = int(height * LINE_POSITION)

# Vehicle classes
vehicle_classes = [1, 2, 3, 5, 7]
# Virtual incident-detection line
line_y = int(height * 0.70)

# Store the previous center position of each vehicle
previous_positions = {}

# Store vehicles that already crossed the line
crossed_vehicles = set()
# Store detected incidents
events = []

# Example location
latitude = 13.0827
longitude = 80.2707

# Event image counter
event_image_counter = 1

print("Incident detection started!")

while True:
    ret, frame = video.read()

    if not ret:
        break

    # Track vehicles in the current frame
    results = vehicle_model.track(
        frame,
        persist=True,
        classes=vehicle_classes,
        conf=0.25,
        verbose=False
    )

    # Draw the virtual line
    cv.line(
        frame,
        (0, line_y),
        (width, line_y),
        (0, 0, 255),
        3
    )

    boxes = results[0].boxes

    if boxes.id is not None:
        track_ids = boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Calculate the center of the vehicle
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Draw vehicle box
            cv.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Draw vehicle center
            cv.circle(
                frame,
                (center_x, center_y),
                5,
                (255, 0, 0),
                -1
            )

            # Check whether this vehicle crossed the line
            if track_id in previous_positions:
                previous_y = previous_positions[track_id]

                crossed_downward = (
                    previous_y < line_y and center_y >= line_y
                )

                crossed_upward = (
                    previous_y > line_y and center_y <= line_y
                )

            if track_id not in crossed_vehicles and center_y >= int(height * 0.30):

                    crossed_vehicles.add(track_id)
                    # Get incident information
                    timestamp_seconds = video.get(cv.CAP_PROP_POS_MSEC) / 1000
                    minutes = int(timestamp_seconds // 60)
                    seconds = int(timestamp_seconds % 60)

                    timestamp = f"{minutes:02d}:{seconds:02d}"

                    # Get vehicle confidence
                    incident_confidence = float(box.conf[0])

                    # Save an image of the incident
                    event_image_name = f"event_{event_image_counter:03d}.jpg"
                    event_image_path = f"ml_person_3/data/{event_image_name}"

                    image_saved = cv.imwrite(event_image_path, frame)

                    print("Incident image save result:", image_saved)
                    print("Incident image path:", event_image_path)

                    # Create the event in the required format
                    event = {
                        "event_type": "traffic_violation",
                        "timestamp": timestamp,
                        "latitude": latitude,
                        "longitude": longitude,
                        "confidence": round(incident_confidence, 2),
                        "severity": "MEDIUM",
                        "image": event_image_name
                    }

                    events.append(event)

                    event_image_counter += 1

                    print(
                                            f"Incident detected! "
                                            f"Vehicle ID: {track_id} crossed the line."
                                        )

                    cv.putText(
                                            frame,
                                            "INCIDENT DETECTED",
                                            (x1, max(y1 - 35, 30)),
                                            cv.FONT_HERSHEY_SIMPLEX,
                                            0.7,
                                            (0, 0, 255),
                                            2
                                        )

# Save the current center position
previous_positions[track_id] = center_y

            # Display tracking ID
cv.putText(
                frame,
                f"ID: {track_id}",
                (x1, y2 + 20),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

    # Save the processed frame
output.write(frame)

video.release()
output.release()
print("Total events detected:", len(events))
# Save all incidents as a JSON file
with open(
    "ml_person_3/data/events.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(events, file, indent=4)

print("Events JSON file saved:")
print("ml_person_3/data/events.json")

print("Incident detection completed!")
print("Output video saved:")
print("ml_person_3/data/incident_output.mp4")