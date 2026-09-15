from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import json


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "Models" / "UVH-26-MV-YOLOv11-S.pt"
IMAGE_PATH = BASE_DIR / "Videos" / "india_traffic.png"

OUTPUT_DIR = BASE_DIR / "Outputs"

EVENTS_JSON_PATH = OUTPUT_DIR / "events.json"
EVENTS_IMAGE_DIR = OUTPUT_DIR / "events"
ANNOTATED_IMAGE_PATH = OUTPUT_DIR / "traffic_detected.jpg"


# ============================================================
# CONGESTION SETTINGS
# ============================================================

# Image-based traffic-density estimation.
#
# Vehicle count is the primary indicator.
# Vehicle occupancy is a secondary supporting factor.
#
# This is a prototype heuristic. It does not determine
# whether vehicles are moving or stationary.

OCCUPANCY_ADJUSTMENT_LIMIT = 10


# ============================================================
# VEHICLE CLASS MAPPING
# ============================================================

class_map = {
    "Hatchback": "Car",
    "Sedan": "Car",
    "SUV": "Car",
    "MUV": "Car",
    "Van": "Car",

    "Three-wheeler": "Auto",

    "Bus": "Bus",
    "Mini-bus": "Bus",

    "Truck": "Truck",
    "LCV": "Truck",
    "tempo-traveller": "Truck",

    "Two-wheeler": "Motorcycle"
}


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nModel file not found:\n{MODEL_PATH}"
    )

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"\nInput image not found:\n{IMAGE_PATH}"
    )


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EVENTS_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("Loading YOLO model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")


# ============================================================
# RUN DETECTION
# ============================================================

print(
    f"\nRunning detection on:"
    f"\n{IMAGE_PATH}"
)

results = model(
    str(IMAGE_PATH),
    conf=0.40
)


# ============================================================
# STORAGE
# ============================================================

detections = []

image_width = 0
image_height = 0


# ============================================================
# PROCESS YOLO RESULTS
# ============================================================

for result in results:

    # --------------------------------------------------------
    # Save annotated image
    # --------------------------------------------------------

    result.save(
        filename=str(ANNOTATED_IMAGE_PATH)
    )

    # --------------------------------------------------------
    # Get image dimensions
    # --------------------------------------------------------

    image_height, image_width = result.orig_shape

    # --------------------------------------------------------
    # Check whether detections exist
    # --------------------------------------------------------

    if result.boxes is None:
        continue

    # --------------------------------------------------------
    # Extract detection information
    # --------------------------------------------------------

    class_ids = result.boxes.cls.tolist()
    confidences = result.boxes.conf.tolist()
    bounding_boxes = result.boxes.xyxy.tolist()

    # --------------------------------------------------------
    # Process every detected object
    # --------------------------------------------------------

    for cls, conf, bbox in zip(
        class_ids,
        confidences,
        bounding_boxes
    ):

        raw_class = model.names[int(cls)]

        final_class = class_map.get(raw_class)

        # Ignore unmapped classes
        if final_class is None:
            continue

        x1, y1, x2, y2 = [
            int(value)
            for value in bbox
        ]

        # ----------------------------------------------------
        # Bounding-box area
        # ----------------------------------------------------

        box_width = max(
            0,
            x2 - x1
        )

        box_height = max(
            0,
            y2 - y1
        )

        box_area = box_width * box_height

        # ----------------------------------------------------
        # Store detection
        # ----------------------------------------------------

        detection = {
            "event_type": final_class.lower(),

            "confidence": round(
                float(conf),
                3
            ),

            "bbox": [
                x1,
                y1,
                x2,
                y2
            ],

            "box_area": box_area
        }

        detections.append(detection)

        print(
            f"{raw_class} -> "
            f"{final_class} "
            f"({float(conf):.2f})"
        )


# ============================================================
# VEHICLE COUNT
# ============================================================

vehicle_count = len(detections)

print("\n----------------------------------------")
print(
    f"Total vehicles detected: "
    f"{vehicle_count}"
)
print("----------------------------------------")


# ============================================================
# CALCULATE VEHICLE IMAGE OCCUPANCY
# ============================================================

total_vehicle_area = sum(
    detection["box_area"]
    for detection in detections
)

if image_width > 0 and image_height > 0:

    image_area = image_width * image_height

    occupancy_percent = (
        total_vehicle_area / image_area
    ) * 100

else:

    occupancy_percent = 0.0


occupancy_percent = round(
    occupancy_percent,
    2
)


print(
    f"Vehicle image occupancy: "
    f"{occupancy_percent:.2f}%"
)


# ============================================================
# CALCULATE BASE DENSITY SCORE
# ============================================================

# Vehicle count is the primary factor.

if vehicle_count <= 4:

    base_score = 10

elif vehicle_count <= 5:

    base_score = 20

elif vehicle_count <= 8:

    base_score = 35

elif vehicle_count <= 9:

    base_score = 40

elif vehicle_count <= 12:

    base_score = 55

elif vehicle_count <= 14:

    base_score = 60

else:

    base_score = 75


# ============================================================
# CALCULATE OCCUPANCY ADJUSTMENT
# ============================================================

# Occupancy is only a supporting factor.
#
# Example:
# 10% occupancy -> approximately +5 points.
#
# Maximum adjustment = 10 points.

occupancy_adjustment = min(
    OCCUPANCY_ADJUSTMENT_LIMIT,
    occupancy_percent / 2
)


# ============================================================
# FINAL CONGESTION SCORE
# ============================================================

congestion_score = min(
    100,
    base_score + occupancy_adjustment
)

congestion_score = round(
    congestion_score,
    2
)


# ============================================================
# DETERMINE CONGESTION LEVEL
# ============================================================

if congestion_score >= 70:

    congestion_severity = "high"

elif congestion_score >= 30:

    congestion_severity = "moderate"

else:

    congestion_severity = "low"


print(
    f"Base density score: "
    f"{base_score:.2f}"
)

print(
    f"Occupancy adjustment: "
    f"{occupancy_adjustment:.2f}"
)

print(
    f"Final traffic-density score: "
    f"{congestion_score:.2f}/100"
)

print(
    f"Traffic-density level: "
    f"{congestion_severity.upper()}"
)


# ============================================================
# TIMESTAMP
# ============================================================

timestamp = datetime.now().isoformat()


# ============================================================
# CREATE CURRENT VEHICLE EVENTS
# ============================================================

current_events = []


for detection in detections:

    event = {
        "event_type": detection["event_type"],

        "confidence": detection["confidence"],

        "bbox": detection["bbox"],

        "timestamp": timestamp,

        "image": "Outputs/traffic_detected.jpg",

        # Prototype location.
        # Replace with real GPS data later.
        "latitude": 13.08,

        "longitude": 80.27
    }

    current_events.append(event)


# ============================================================
# CREATE DERIVED CONGESTION EVENT
# ============================================================

congestion_event = {
    "event_type": "congestion",

    "vehicle_count": vehicle_count,

    "vehicle_occupancy_percent": occupancy_percent,

    "congestion_score": congestion_score,

    "severity": congestion_severity,

    "timestamp": timestamp,

    "image": "Outputs/traffic_detected.jpg",

    "latitude": 13.08,

    "longitude": 80.27,

    "detection_method":
        "image_based_vehicle_density_estimation"
}


# ============================================================
# LOAD EXISTING EVENTS.JSON
# ============================================================

existing_events = []


if EVENTS_JSON_PATH.exists():

    try:

        with open(
            EVENTS_JSON_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

            if isinstance(data, list):

                existing_events = data

            else:

                print(
                    "\nWarning: existing events.json "
                    "does not contain a JSON list."
                )

    except json.JSONDecodeError:

        print(
            "\nWarning: existing events.json "
            "contains invalid JSON."
        )

        print(
            "Starting with a fresh event list."
        )


# ============================================================
# REMOVE PREVIOUS TRAFFIC RESULTS
# ============================================================

traffic_event_types = {
    "car",
    "motorcycle",
    "auto",
    "bus",
    "truck",
    "congestion"
}


existing_events = [
    event
    for event in existing_events
    if event.get("event_type")
    not in traffic_event_types
]


# ============================================================
# COMBINE EVENTS
# ============================================================

final_events = []

final_events.extend(existing_events)

final_events.extend(current_events)

final_events.append(congestion_event)


# ============================================================
# WRITE EVENTS.JSON
# ============================================================

with open(
    EVENTS_JSON_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        final_events,
        file,
        indent=4
    )


# ============================================================
# FINAL STATUS
# ============================================================

print("\n========================================")
print("Detection complete!")
print("========================================")

print(
    f"\nAnnotated image saved to:"
    f"\n{ANNOTATED_IMAGE_PATH}"
)

print(
    f"\nEvents JSON saved to:"
    f"\n{EVENTS_JSON_PATH}"
)

print(
    f"\nVehicles detected: "
    f"{vehicle_count}"
)

print(
    f"Vehicle occupancy: "
    f"{occupancy_percent:.2f}%"
)

print(
    f"Congestion score: "
    f"{congestion_score:.2f}/100"
)

print(
    f"Congestion: "
    f"{congestion_severity.upper()}"
)

print(
    "\nVehicle detections and image-based "
    "congestion estimate saved successfully."
)