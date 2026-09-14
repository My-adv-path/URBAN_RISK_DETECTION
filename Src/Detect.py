from ultralytics import YOLO

# Load the pretrained YOLO model
model = YOLO("yolo26n.pt")

# Run detection on the traffic image
results = model("Videos/traffic.jpg")

# Save the image with bounding boxes and labels
for result in results:
    result.save(filename="Outputs/traffic_detected.jpg")

print("Detection complete!")
print("Annotated image saved to Outputs/traffic_detected.jpg")