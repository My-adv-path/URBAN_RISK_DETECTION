from ultralytics import YOLO

MODEL_PATH = "Models/UVH-26-MV-YOLOv11-S.pt"
VIDEO_PATH = "Videos/indian_traffic.mp4"

model = YOLO(MODEL_PATH)

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

results = model.predict(
    source=VIDEO_PATH,
    conf=0.40,
    save=True,
    stream=True
)

frame_number = 0

for result in results:
    frame_number += 1

    counts = {
        "car": 0,
        "auto": 0,
        "bus": 0,
        "truck": 0,
        "motorcycle": 0
    }

    if result.boxes is not None:
        for class_id in result.boxes.cls.int().cpu().tolist():
            raw_class = result.names[class_id]
            vehicle_type = class_map.get(raw_class)

            if vehicle_type is not None:
                counts[vehicle_type] += 1

    print(
        f"Frame {frame_number}: "
        f"car={counts['car']} "
        f"auto={counts['auto']} "
        f"bus={counts['bus']} "
        f"truck={counts['truck']} "
        f"motorcycle={counts['motorcycle']}"
    )

print("Video detection complete!")