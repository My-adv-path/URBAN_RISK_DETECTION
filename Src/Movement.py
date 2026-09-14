import cv2
import math
from collections import defaultdict, deque
from ultralytics import YOLO


# ==========================================
# SETTINGS
# ==========================================

VIDEO_PATH = "Videos/traffic.mp4"
TRACKER_CONFIG = "configs/botsort_custom.yaml"

# Load YOLO model
model = YOLO("yolo26n.pt")


# ==========================================
# VEHICLE CLASSES
# ==========================================

VEHICLE_CLASSES = {
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle"
}


# ==========================================
# MOVEMENT SETTINGS
# ==========================================

# Minimum center movement considered meaningful.
CENTER_MOVEMENT_THRESHOLD = 2.0

# Minimum relative bounding-box area change
# considered significant.
AREA_CHANGE_THRESHOLD = 0.015

# Number of recent frames used for smoothing.
HISTORY_LENGTH = 10

# Number of consecutive frames required
# before changing movement state.
STATE_CHANGE_FRAMES = 8


# ==========================================
# OPEN VIDEO
# ==========================================

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


# ==========================================
# DISPLAY WINDOW
# ==========================================

cv2.namedWindow(
    "ML-2 Movement Detection",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "ML-2 Movement Detection",
    1000,
    600
)


# ==========================================
# TRACK HISTORY
# ==========================================

# Previous center position for every track
previous_positions = {}

# Previous bounding-box area for every track
previous_areas = {}

# Recent center movement values
movement_history = defaultdict(
    lambda: deque(maxlen=HISTORY_LENGTH)
)

# Recent bounding-box area-change values
area_change_history = defaultdict(
    lambda: deque(maxlen=HISTORY_LENGTH)
)


# ==========================================
# MOVEMENT STATE
# ==========================================

# Current stable state:
# MOVING or STATIONARY
movement_state = {}

# Number of consecutive frames suggesting
# a different state
state_change_counter = defaultdict(int)


# ==========================================
# MAIN VIDEO LOOP
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        break


    # ======================================
    # YOLO + CUSTOM BoT-SORT
    # ======================================

    results = model.track(
        frame,
        persist=True,

        # Use our custom BoT-SORT settings
        tracker=TRACKER_CONFIG,

        conf=0.35,

        verbose=False
    )

    result = results[0]


    # ======================================
    # PROCESS TRACKED OBJECTS
    # ======================================

    if result.boxes.id is not None:

        track_ids = (
            result.boxes.id
            .int()
            .cpu()
            .tolist()
        )

        class_ids = (
            result.boxes.cls
            .int()
            .cpu()
            .tolist()
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .tolist()
        )

        boxes = (
            result.boxes.xyxy
            .cpu()
            .tolist()
        )


        for track_id, class_id, confidence, box in zip(
            track_ids,
            class_ids,
            confidences,
            boxes
        ):

            class_name = result.names[class_id]


            # ==================================
            # ONLY PROCESS VEHICLES
            # ==================================

            if class_name not in VEHICLE_CLASSES:
                continue


            x1, y1, x2, y2 = box


            # ==================================
            # CENTER POINT
            # ==================================

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )


            # ==================================
            # BOUNDING-BOX AREA
            # ==================================

            box_width = max(
                1,
                x2 - x1
            )

            box_height = max(
                1,
                y2 - y1
            )

            current_area = (
                box_width * box_height
            )


            # ==================================
            # CENTER MOVEMENT
            # ==================================

            center_movement = 0.0

            if track_id in previous_positions:

                previous_x, previous_y = (
                    previous_positions[track_id]
                )

                center_movement = math.sqrt(
                    (center_x - previous_x) ** 2
                    +
                    (center_y - previous_y) ** 2
                )


            # ==================================
            # BOUNDING-BOX AREA CHANGE
            # ==================================

            area_change = 0.0

            if track_id in previous_areas:

                previous_area = (
                    previous_areas[track_id]
                )

                if previous_area > 0:

                    area_change = (
                        current_area
                        - previous_area
                    ) / previous_area


            # ==================================
            # SAVE CURRENT VALUES
            # ==================================

            previous_positions[track_id] = (
                center_x,
                center_y
            )

            previous_areas[track_id] = (
                current_area
            )


            # ==================================
            # STORE HISTORY
            # ==================================

            movement_history[track_id].append(
                center_movement
            )

            area_change_history[track_id].append(
                area_change
            )


            # ==================================
            # SMOOTH MOVEMENT VALUES
            # ==================================

            recent_movements = (
                movement_history[track_id]
            )

            recent_area_changes = (
                area_change_history[track_id]
            )


            average_center_movement = (
                sum(recent_movements)
                / len(recent_movements)
            )

            average_area_change = (
                sum(recent_area_changes)
                / len(recent_area_changes)
            )


            # ==================================
            # DETERMINE OBSERVED MOVEMENT
            # ==================================

            center_is_moving = (
                average_center_movement
                > CENTER_MOVEMENT_THRESHOLD
            )

            box_is_changing = (
                abs(average_area_change)
                > AREA_CHANGE_THRESHOLD
            )


            if (
                center_is_moving
                or box_is_changing
            ):

                observed_state = "MOVING"

            else:

                observed_state = "STATIONARY"


            # ==================================
            # INITIALIZE STATE
            # ==================================

            if track_id not in movement_state:

                movement_state[track_id] = (
                    observed_state
                )

                state_change_counter[track_id] = 0


            # ==================================
            # STATE STABILITY / HYSTERESIS
            # ==================================

            current_state = movement_state[
                track_id
            ]


            if observed_state == current_state:

                # The object agrees with its
                # current state, so reset the
                # possible-change counter.
                state_change_counter[track_id] = 0


            else:

                # The object appears to be changing
                # state. Require multiple consecutive
                # observations before accepting it.
                state_change_counter[track_id] += 1


                if (
                    state_change_counter[track_id]
                    >= STATE_CHANGE_FRAMES
                ):

                    movement_state[track_id] = (
                        observed_state
                    )

                    state_change_counter[track_id] = 0


            # Final stable state
            final_state = movement_state[
                track_id
            ]


            # ==================================
            # DRAW BOUNDING BOX
            # ==================================

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (255, 255, 255),
                2
            )


            # ==================================
            # DRAW CENTER POINT
            # ==================================

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )


            # ==================================
            # DRAW LABEL
            # ==================================

            label = (
                f"{class_name} "
                f"ID:{track_id} "
                f"{final_state}"
            )


            cv2.putText(
                frame,
                label,
                (
                    int(x1),
                    max(
                        20,
                        int(y1) - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )


            # ==================================
            # DISPLAY MOTION VALUES
            # ==================================

            movement_text = (
                f"Move: "
                f"{average_center_movement:.2f}"
                f"  "
                f"Area: "
                f"{average_area_change * 100:.1f}%"
            )


            cv2.putText(
                frame,
                movement_text,
                (
                    int(x1),
                    int(y2) + 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )


    # ======================================
    # TITLE
    # ======================================

    cv2.putText(
        frame,
        "Movement Analysis",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "Custom BoT-SORT + Motion + Box Scale",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ======================================
    # SHOW VIDEO
    # ======================================

    cv2.imshow(
        "ML-2 Movement Detection",
        frame
    )


    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()
cv2.destroyAllWindows()

print("\nMovement analysis complete.")