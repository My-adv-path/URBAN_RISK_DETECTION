from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolo26n.pt")

# Track objects throughout the video
results = model.track(
    source="Videos/traffic.mp4",
    tracker="bytetrack.yaml",
    conf=0.35,
    save=True,
    stream=True
)

frame_number = 0

for result in results:
    frame_number += 1

    print(f"\nFrame {frame_number}")

    # Check whether tracking IDs exist
    if result.boxes.id is not None:

        track_ids = result.boxes.id.int().cpu().tolist()
        class_ids = result.boxes.cls.int().cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()

        for track_id, class_id, confidence in zip(
            track_ids,
            class_ids,
            confidences
        ):
            class_name = result.names[class_id]

            print(
                f"ID={track_id} | "
                f"Class={class_name} | "
                f"Confidence={confidence:.2f}"
            )

print("\nTracking complete!")