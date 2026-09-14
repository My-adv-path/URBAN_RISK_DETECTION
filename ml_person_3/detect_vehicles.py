
from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("yolo11n.pt")

# Detect objects in our road image
results = model("ml_person_3/data/road.jpg", save=True)

print("Detection completed!")
print("The result image has been saved.")