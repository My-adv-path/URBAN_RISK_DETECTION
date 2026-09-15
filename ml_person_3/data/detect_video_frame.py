from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Detect objects in the clean video frame
results = model(
    "ml_person_3/data/frame_10_seconds.jpg",
    save=True
)

# Print detected objects
for box in results[0].boxes:
    class_id = int(box.cls[0])
    object_name = results[0].names[class_id]
    confidence = float(box.conf[0])

    print(f"Detected: {object_name}, Confidence: {confidence:.2f}")

print("Detection completed!")
print("The result image has been saved.")