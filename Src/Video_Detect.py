from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolo26n.pt")

# Run detection on the traffic video
results = model.predict(
    source="Videos/traffic.mp4",
    conf=0.35,
    save=True,
    stream=True
)

# Process each frame
frame_number = 0

for result in results:
    frame_number += 1

    print(
        f"Frame {frame_number}: "
        f"{len(result.boxes)} objects detected"
    )

print("Video detection complete!")