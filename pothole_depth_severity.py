import cv2
import torch
import json
import numpy as np
from datetime import datetime, timezone
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline, AutoImageProcessor


# ============================================================
# SETTINGS
# ============================================================

VIDEO_PATH = "testing_video.mp4"
YOLO_MODEL = "best_new.pt"

# The class name in your YOLO model that represents a pothole.
# This MUST exactly match one of the names printed under
# "YOLO classes:" below (case-sensitive).
POTHOLE_CLASS_NAME = "D40"

# Run depth estimation once every N frames after a pothole
# is first seen (roughly every 8 seconds at 30 FPS)
DEPTH_INTERVAL = 240

# YOLO confidence
YOLO_CONF = 0.30

# YOLO image size
YOLO_IMGSZ = 640

# Print every class detected in every frame (debug mode).
# Turn this on if depth still never triggers, to see what
# your model is actually outputting.
DEBUG_PRINT_ALL_DETECTIONS = True

# Only compute/send depth for detections at least this confident.
# Weak detections tend to have noisy/misaligned boxes, which
# produces unreliable depth (0.0 cm or wild spikes).
MIN_CONFIDENCE_FOR_EVENT = 0.50

# Minimum relative depth (cm) below which we don't bother
# generating an event - filters out flat/noise detections.
MIN_DEPTH_CM_FOR_EVENT = 0.5

# Final events always report depth in this 0-10 range.
OUTPUT_DEPTH_MIN_CM = 0.0
OUTPUT_DEPTH_MAX_CM = 10.0

# Raw (uncalibrated) depth at/above this is treated as the
# "deepest" a pothole can be, and maps to OUTPUT_DEPTH_MAX_CM.
# Raw values below this scale down proportionally. Tune this
# based on the worst-case raw readings you actually see.
ASSUMED_MAX_RAW_DEPTH_CM = 30.0

# Where events get saved as JSON. Written incrementally, so
# partial results are kept even if the script is stopped early.
EVENTS_OUTPUT_PATH = "pothole_events_final.json"


# ============================================================
# CHECK DEVICE
# ============================================================

if torch.cuda.is_available():
    print("CUDA available - using GPU")
    print("GPU:", torch.cuda.get_device_name(0))
    device = 0
else:
    print("CUDA not available - using CPU")
    device = "cpu"


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("Loading YOLO model...")
yolo_model = YOLO(YOLO_MODEL)
print("YOLO loaded.")

print("YOLO classes:")
print(yolo_model.names)

# --------------------------------------------------------
# SANITY CHECK: does POTHOLE_CLASS_NAME actually exist?
# --------------------------------------------------------

available_names = list(yolo_model.names.values())

if POTHOLE_CLASS_NAME not in available_names:
    print(
        f"\n*** WARNING: '{POTHOLE_CLASS_NAME}' is NOT in the model's "
        f"class list: {available_names}\n"
        f"*** Depth estimation will NEVER trigger until you fix "
        f"POTHOLE_CLASS_NAME to match one of the names above.\n"
    )


# ============================================================
# LOAD DEPTH MODEL
# ============================================================

print("Loading depth estimation model...")

depth_pipe = pipeline(
    task="depth-estimation",
    model="Intel/zoedepth-nyu-kitti",
    device=device
)

print("Depth model loaded.")

depth_processor = AutoImageProcessor.from_pretrained(
    "Intel/zoedepth-nyu-kitti"
)

print("Depth processor loaded.")


# ============================================================
# FUNCTION TO CALCULATE POTHOLE DEPTH
# ============================================================

def calculate_pothole_depth(depth_map, x1, y1, x2, y2, other_boxes=None):
    """
    Estimate relative pothole depth by comparing:
    1. Depth inside the pothole bounding box
    2. Depth of the surrounding road

    other_boxes: list of (bx1, by1, bx2, by2) for OTHER pothole
    detections in the same frame. These get excluded from the
    "surrounding road" sample so a neighboring pothole doesn't
    get mistaken for flat road and skew the baseline.
    """

    h, w = depth_map.shape

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    pothole_region = depth_map[y1:y2, x1:x2]

    if pothole_region.size == 0:
        return None

    p10 = np.percentile(pothole_region, 10)
    p90 = np.percentile(pothole_region, 90)

    pothole_valid = pothole_region[
        (pothole_region >= p10) &
        (pothole_region <= p90)
    ]

    if pothole_valid.size == 0:
        return None

    # Higher depth = farther from camera
    pothole_bottom_depth = float(np.percentile(pothole_valid, 80))

    box_width = x2 - x1
    box_height = y2 - y1

    margin_x = max(15, int(box_width * 0.40))
    margin_y = max(15, int(box_height * 0.40))

    sx1 = max(0, x1 - margin_x)
    sy1 = max(0, y1 - margin_y)
    sx2 = min(w, x2 + margin_x)
    sy2 = min(h, y2 + margin_y)

    surrounding = depth_map[sy1:sy2, sx1:sx2]

    if surrounding.size == 0:
        return None

    mask = np.ones(surrounding.shape, dtype=bool)

    inner_x1 = x1 - sx1
    inner_y1 = y1 - sy1
    inner_x2 = x2 - sx1
    inner_y2 = y2 - sy1

    mask[inner_y1:inner_y2, inner_x1:inner_x2] = False

    # --------------------------------------------------------
    # ALSO EXCLUDE OTHER POTHOLES FROM THE "ROAD" SAMPLE
    #
    # Prevents a neighboring pothole inside the margin from
    # being mistaken for flat road, which would push the
    # baseline too far away and inflate the depth result.
    # --------------------------------------------------------

    if other_boxes:

        for (ox1, oy1, ox2, oy2) in other_boxes:

            # Convert to local (surrounding-region) coordinates
            local_x1 = max(0, ox1 - sx1)
            local_y1 = max(0, oy1 - sy1)
            local_x2 = min(surrounding.shape[1], ox2 - sx1)
            local_y2 = min(surrounding.shape[0], oy2 - sy1)

            if local_x2 > local_x1 and local_y2 > local_y1:
                mask[local_y1:local_y2, local_x1:local_x2] = False

    road_region = surrounding[mask]

    if road_region.size == 0:
        return None

    r10 = np.percentile(road_region, 10)
    r90 = np.percentile(road_region, 90)

    road_valid = road_region[
        (road_region >= r10) &
        (road_region <= r90)
    ]

    if road_valid.size == 0:
        return None

    road_surface_depth = float(np.median(road_valid))

    relative_depth = pothole_bottom_depth - road_surface_depth

    # Negative means the model did not see the
    # pothole bottom as farther than the road.
    if relative_depth < 0:
        relative_depth = 0.0

    return (road_surface_depth, pothole_bottom_depth, relative_depth)

# ============================================================
# CALCULATE POTHOLE SEVERITY
# ============================================================

def calculate_severity(depth_cm):
    """
    Calculate pothole severity based on the scaled pothole depth.

    Depth range:
        0 - <3 cm   -> Low
        3 - 6 cm    -> Medium
        >6 cm       -> High
    """

    if depth_cm < 3.0:
        return "Low"
    elif depth_cm <= 6.0:
        return "Medium"
    else:
        return "High"
# ============================================================
# SCALE RAW DEPTH INTO THE OUTPUT 0-10 RANGE
# ============================================================

def scale_depth_cm(raw_depth_cm):
    """
    Proportionally maps a raw (uncalibrated) depth in cm into
    the [OUTPUT_DEPTH_MIN_CM, OUTPUT_DEPTH_MAX_CM] range, using
    ASSUMED_MAX_RAW_DEPTH_CM as the reference ceiling.

    raw_depth_cm = 0                      -> OUTPUT_DEPTH_MIN_CM
    raw_depth_cm = ASSUMED_MAX_RAW_DEPTH_CM -> OUTPUT_DEPTH_MAX_CM
    raw_depth_cm > ASSUMED_MAX_RAW_DEPTH_CM -> capped at OUTPUT_DEPTH_MAX_CM
    """

    ratio = raw_depth_cm / ASSUMED_MAX_RAW_DEPTH_CM
    ratio = min(1.0, max(0.0, ratio))

    scaled = OUTPUT_DEPTH_MIN_CM + ratio * (OUTPUT_DEPTH_MAX_CM - OUTPUT_DEPTH_MIN_CM)

    return scaled


# ============================================================
# HELPER: RUN DEPTH ESTIMATION ON A FRAME
# ============================================================

def run_depth_estimation(frame):

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    inputs = depth_processor(images=pil_image, return_tensors="pt")

    inputs = {
        key: value.to(depth_pipe.model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = depth_pipe.model(**inputs)

    depth_map = outputs.predicted_depth

    depth_map = (
        depth_map
        .squeeze()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    print(
        "RAW MODEL DEPTH:",
        "min =", depth_map.min(),
        "max =", depth_map.max(),
        "mean =", depth_map.mean()
    )

    depth_map = cv2.resize(
        depth_map,
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_CUBIC
    )

    return depth_map


# ============================================================
# GPS PLACEHOLDER
#
# Replace this with your real GPS source, e.g.:
#   - pynmea2 reading from a serial NEO-6M module
#   - a phone GPS logger writing to a file/socket you read here
#   - GPS metadata pulled from the video file itself
#
# Must return a (latitude, longitude) tuple.
# ============================================================

def get_current_gps():

    # DUMMY VALUES - replace with real GPS reading
    latitude = 13.0827
    longitude = 80.2707

    return latitude, longitude


# ============================================================
# EVENT GENERATION
#
# Follows the pipeline:
# Detection -> Bounding Box -> Confidence Score ->
# Timestamp -> GPS Coordinates -> Event Generation
#
# Returns a single dict ready to be sent to the dashboard API.
# ============================================================

def generate_event(
    frame_count,
    class_name,
    confidence,
    x1, y1, x2, y2,
    depth_cm
):

    latitude, longitude = get_current_gps()
    severity = calculate_severity(depth_cm)

    event = {
        "event_type": "pothole_detected",
        "class": class_name,
        "confidence": round(confidence, 3),
        "bounding_box": {
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2)
        },
        "depth_cm": round(depth_cm, 1),
        "severity": severity,
        "severity_score": round(depth_cm, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gps": {
            "latitude": latitude,
            "longitude": longitude
        },
        "frame": frame_count
    }

    return event


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Processing video... ({total_frames} frames)")


# ============================================================
# VARIABLES
# ============================================================

frame_count = 0
depth_map = None

# All generated events, saved to EVENTS_OUTPUT_PATH as JSON.
all_events = []

# Frame number on which depth was last computed.
# Starting negative forces the very first pothole detection
# to trigger depth estimation immediately, instead of waiting
# for frame_count to be an exact multiple of DEPTH_INTERVAL.
last_depth_frame = -DEPTH_INTERVAL


def save_events_to_json():
    with open(EVENTS_OUTPUT_PATH, "w") as f:
        json.dump(all_events, f, indent=2)


# ============================================================
# VIDEO LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # ========================================================
    # YOLO DETECTION
    # ========================================================

    results = yolo_model.predict(
        frame,
        conf=YOLO_CONF,
        imgsz=YOLO_IMGSZ,
        verbose=False,
        device=device
    )

    result = results[0]

    # ========================================================
    # DEBUG: PRINT EVERY DETECTION THIS FRAME
    # ========================================================

    if DEBUG_PRINT_ALL_DETECTIONS and len(result.boxes) > 0:

        detected_names = [
            yolo_model.names[int(box.cls[0])]
            for box in result.boxes
        ]

        print(f"[Frame {frame_count}] Detected: {detected_names}")

    # ========================================================
    # CHECK WHETHER POTHOLE IS DETECTED
    # ========================================================

    pothole_detected = False

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = yolo_model.names[class_id]

        if class_name == POTHOLE_CLASS_NAME:
            pothole_detected = True
            break

    # ========================================================
    # DEPTH ESTIMATION
    #
    # Runs when:
    # 1. A pothole exists in this frame, AND
    # 2. Either we have never run depth yet, or at least
    #    DEPTH_INTERVAL frames have passed since the last run.
    # ========================================================

    if pothole_detected and (frame_count - last_depth_frame >= DEPTH_INTERVAL):

        print(f"Running depth estimation at frame {frame_count}...")

        depth_map = run_depth_estimation(frame)
        last_depth_frame = frame_count

        print("Depth estimation completed.")

    # ========================================================
    # COLLECT ALL POTHOLE BOXES IN THIS FRAME
    #
    # Needed so each box's depth calculation can exclude the
    # OTHER pothole boxes from its "surrounding road" sample.
    # ========================================================

    pothole_boxes_this_frame = []

    for box in result.boxes:

        class_id = int(box.cls[0])
        class_name = yolo_model.names[class_id]

        if class_name != POTHOLE_CLASS_NAME:
            continue

        bx1, by1, bx2, by2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )

        pothole_boxes_this_frame.append((int(bx1), int(by1), int(bx2), int(by2)))

    # ========================================================
    # PROCESS DETECTED OBJECTS -> PRINT DEPTH TO TERMINAL
    # ========================================================

    for box_index, box in enumerate(result.boxes):

        class_id = int(box.cls[0])
        class_name = yolo_model.names[class_id]

        if class_name != POTHOLE_CLASS_NAME:
            continue

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )

        confidence = float(box.conf[0])

        if confidence < MIN_CONFIDENCE_FOR_EVENT:
            print(f"[Frame {frame_count}] {class_name} conf={confidence:.2f} | below confidence threshold, skipping.")
            continue

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1] - 1, x2)
        y2 = min(frame.shape[0] - 1, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        if depth_map is None:
            print(f"[Frame {frame_count}] {class_name} conf={confidence:.2f} | depth not yet available")
            continue

        current_box = (int(x1), int(y1), int(x2), int(y2))

        other_boxes = [
            b for b in pothole_boxes_this_frame
            if b != current_box
        ]

        depth_result = calculate_pothole_depth(depth_map, x1, y1, x2, y2, other_boxes=other_boxes)

        if depth_result is None:
            print(f"[Frame {frame_count}] {class_name} conf={confidence:.2f} | depth: N/A")
            continue

        road_surface_depth, pothole_bottom_depth, relative_depth = depth_result

        # NOTE: this is still an UNCALIBRATED estimate.
        relative_depth_cm = relative_depth * 100

        print(
            f"[Frame {frame_count}] {class_name} conf={confidence:.2f} | "
            f"Road surface: {road_surface_depth:.3f} m | "
            f"Pothole bottom: {pothole_bottom_depth:.3f} m | "
            f"Pothole depth: {relative_depth:.3f} m | "
            f"Approx: {relative_depth_cm:.1f} cm"
        )

        # ====================================================
        # EVENT GENERATION
        # ====================================================

        if relative_depth_cm < MIN_DEPTH_CM_FOR_EVENT:
            print(f"[Frame {frame_count}] Depth below threshold, skipping event.")
            continue

        scaled_depth_cm = scale_depth_cm(relative_depth_cm)
        severity = calculate_severity(scaled_depth_cm)


        if relative_depth_cm > ASSUMED_MAX_RAW_DEPTH_CM:
            print(
                f"[Frame {frame_count}] *** raw {relative_depth_cm:.1f} cm capped "
                f"(exceeds assumed max {ASSUMED_MAX_RAW_DEPTH_CM:.1f} cm) "
                f"-> scaled {scaled_depth_cm:.1f} cm ***"
            )
        else:
            print(
                f"[Frame {frame_count}] raw {relative_depth_cm:.1f} cm "
                f"-> scaled {scaled_depth_cm:.1f} cm"
            )

        event = generate_event(
            frame_count=frame_count,
            class_name=class_name,
            confidence=confidence,
            x1=x1, y1=y1, x2=x2, y2=y2,
            depth_cm=scaled_depth_cm
        )

        print(f"[Frame {frame_count}] EVENT:", event)

        all_events.append(event)
        save_events_to_json()

        # Next step: send `event` to the dashboard API here
        # (we'll wire this up once depth output looks correct).


# ============================================================
# CLEANUP
# ============================================================

cap.release()

save_events_to_json()
print(f"Saved {len(all_events)} event(s) to {EVENTS_OUTPUT_PATH}")

print("Processing completed.")
