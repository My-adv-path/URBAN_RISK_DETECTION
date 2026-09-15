import cv2 as cv
import easyocr

# Create OCR reader
reader = easyocr.Reader(["en"], gpu=False)

# Read an image
image = cv.imread("ml_person_3/data/road.jpg")

if image is None:
    print("Image could not be opened")
    exit()

# Run OCR
results = reader.readtext(
    image,
    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)

print("OCR Results:")

for result in results:
    text = result[1]
    confidence = result[2]

    print(f"Text: {text}, Confidence: {confidence:.2f}")

print("ANPR test completed!")