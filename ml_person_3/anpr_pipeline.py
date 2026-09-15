import cv2 as cv
import numpy as np
import re
import time
import easyocr

from ultralytics import YOLO

# Load vehicle detection model
vehicle_model = YOLO("yolo11n.pt")
plate_model = YOLO("ml_person_3/license_plate_detector.pt")

# EasyOCR reader
reader = easyocr.Reader(["en"], gpu=False)

# Open traffic video
video = cv.VideoCapture("ml_person_3/data/anpr_test.mp4")

if not video.isOpened():
    print("Video cannot be accessed")
    exit()

print("ANPR pipeline started successfully!")

# Move to approximately 10 seconds into the video
video.set(cv.CAP_PROP_POS_MSEC,0)

ret, frame = video.read()


if not ret:
    print("Could not read the video frame")
    video.release()
    exit()

print("First video frame read successfully!")
print("Frame size:", frame.shape)

# Detect vehicles in the first frame
vehicle_classes = [1, 2, 3, 5, 7]

results = vehicle_model.predict(
    frame,
    classes=vehicle_classes,
    conf=0.15
)

print("Vehicles and license plates detected:")

for box in results[0].boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    object_name = results[0].names[class_id]

    # Vehicle bounding box coordinates
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Draw vehicle box
    cv.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Write vehicle name and confidence
    cv.putText(
        frame,
        f"{object_name} {confidence:.2f}",
        (x1, y1 - 10),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    print(
        f"Vehicle: {object_name}, "
        f"Confidence: {confidence:.2f}"
    )

    # Crop the vehicle from the original frame
    vehicle_crop = frame[y1:y2, x1:x2]

    if vehicle_crop.size == 0:
        continue

    # Detect license plates inside the vehicle crop
    plate_results = plate_model.predict(
        vehicle_crop,
        conf=0.25,
        verbose=False
    )
    print(
    "Plates detected in this vehicle:",
    len(plate_results[0].boxes)
)

    for plate_box in plate_results[0].boxes:
        # Plate coordinates relative to the vehicle crop
        px1, py1, px2, py2 = map(
            int,
            plate_box.xyxy[0]
        )

        # Convert plate coordinates to original frame coordinates
        abs_x1 = x1 + px1
        abs_y1 = y1 + py1
        abs_x2 = x1 + px2
        abs_y2 = y1 + py2

        # Crop the license plate
        plate_crop = frame[abs_y1:abs_y2, abs_x1:abs_x2]

        if plate_crop.size == 0:
            continue

        # Read text from the license plate
        ocr_results = reader.readtext(plate_crop)

        plate_text = ""

        for detection in ocr_results:
            detected_text = detection[1]
            plate_text += detected_text + " "

        # Keep only letters and numbers
        plate_text = re.sub(
            r"[^A-Za-z0-9]",
            "",
            plate_text
        ).upper()

        # Draw license plate box
        cv.rectangle(
            frame,
            (abs_x1, abs_y1),
            (abs_x2, abs_y2),
            (0, 0, 255),
            2
        )

        # Display detected plate text
        cv.putText(
            frame,
            plate_text if plate_text else "Plate",
            (abs_x1, abs_y1 - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        print(
            "License plate detected:",
            plate_text if plate_text else "Text not readable"
        )

# Save the annotated frame
cv.imwrite(
    "ml_person_3/data/vehicle_detection_test.jpg",
    frame
)

print("Annotated vehicle image saved!")
video.release()

print("Vehicle detection test completed!")