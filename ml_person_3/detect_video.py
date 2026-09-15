from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Run detection on the entire video
results = model.predict(
    source="ml_person_3/data/traffic.mp4",
    save=True,
    stream=True,
    conf=0.25
)

frame_count = 0

for result in results:
    frame_count += 1

    if frame_count % 100 == 0:
        print(f"Processed {frame_count} frames")

print("Video detection completed!")
print("The annotated video has been saved.")