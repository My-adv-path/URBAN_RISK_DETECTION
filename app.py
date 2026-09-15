from ultralytics import YOLO
import cv2
import os

# ============================================
# PATHS
# ============================================

MODEL_PATH = "models/best.pt"
IMAGE_PATH = "input/image.png"
OUTPUT_PATH = "output/detected.jpg"

# ============================================
# LOAD MODEL
# ============================================

print("=" * 50)
print("Loading trained YOLO model...")
print("=" * 50)

model = YOLO(MODEL_PATH)

print("✅ MODEL LOADED SUCCESSFULLY")
print("Classes:", model.names)
print("=" * 50)

# ============================================
# CHECK IMAGE
# ============================================

if not os.path.exists(IMAGE_PATH):
    print(f"❌ Image not found: {IMAGE_PATH}")
    exit()

# ============================================
# RUN DETECTION
# ============================================

print("\n🔍 Running pedestrian detection...")

results = model.predict(
    source=IMAGE_PATH,
    conf=0.25,
    imgsz=640,
    save=False
)

# ============================================
# DRAW RESULTS
# ============================================

result = results[0]

annotated = result.plot()

# ============================================
# SAVE OUTPUT
# ============================================

os.makedirs("output", exist_ok=True)

cv2.imwrite(OUTPUT_PATH, annotated)

print("\n" + "=" * 50)
print("✅ DETECTION COMPLETED")
print("=" * 50)

print(f"Input : {IMAGE_PATH}")
print(f"Output: {OUTPUT_PATH}")

# ============================================
# DETECTION SUMMARY
# ============================================

if result.boxes is not None:

    boxes = result.boxes

    print("\nDetected objects:")

    for i, box in enumerate(boxes):

        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = model.names[cls_id]

        print(
            f"{i + 1}. {class_name} "
            f"({confidence * 100:.2f}%)"
        )

else:

    print("\n⚠️ No objects detected.")

# ============================================
# DISPLAY IMAGE
# ============================================

cv2.imshow("Pedestrian Detection", annotated)

print("\nPress any key on the image window to close.")

cv2.waitKey(0)
cv2.destroyAllWindows()