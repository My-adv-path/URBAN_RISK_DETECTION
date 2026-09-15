from ultralytics import YOLO

model = YOLO("../Models/UVH-26-MV-YOLOv11-S.pt")

results = model("Videos/india_traffic.png", conf=0.40)

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

for result in results:
    for cls, conf in zip(
        result.boxes.cls.tolist(),
        result.boxes.conf.tolist()
    ):
        raw_class = model.names[int(cls)]
        final_class = class_map.get(raw_class)

        if final_class:
            print(f"{raw_class} -> {final_class} ({conf:.2f})")

    result.save(filename="Outputs/traffic_detected.jpg")

print("Detection complete!")
print("Annotated image saved to Outputs/traffic_detected.jpg")