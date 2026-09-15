from ultralytics import YOLO

model = YOLO("Models/UVH-26-MV-YOLOv11-S.pt")

class_map = {
    "Hatchback": "car",
    "Sedan": "car",
    "SUV": "car",
    "MUV": "car",
    "Van": "car",
    "Three-wheeler": "auto",
    "Bus": "bus",
    "Mini-bus": "bus",
    "Truck": "truck",
    "LCV": "truck",
    "tempo-traveller": "truck",
    "Two-wheeler": "motorcycle"
}

results = model.track(
    source="Videos/indian_traffic.mp4",
    tracker="bytetrack.yaml",
    conf=0.40,
    save=True,
    stream=True
)

frame_number = 0

for result in results:
    frame_number += 1

    print(f"\nFrame {frame_number}")

    if result.boxes is not None and result.boxes.id is not None:
        track_ids = result.boxes.id.int().cpu().tolist()
        class_ids = result.boxes.cls.int().cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()

        for track_id, class_id, confidence in zip(
            track_ids,
            class_ids,
            confidences
        ):
            raw_class = result.names[int(class_id)]
            class_name = class_map.get(raw_class)

            if class_name is None:
                continue

            print(
                f"ID={track_id} | "
                f"Class={class_name} | "
                f"Confidence={confidence:.2f}"
            )

print("\nTracking complete!")