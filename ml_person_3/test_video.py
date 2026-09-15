import cv2

video_path = "ml_person_3/data/traffic.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open the video.")
else:
    print("Video opened successfully!")

    # Move to the 10-second position
    cap.set(cv2.CAP_PROP_POS_MSEC, 10000)

    ret, frame = cap.read()

    if ret:
        cv2.imwrite("ml_person_3/data/frame_10_seconds.jpg", frame)
        print("Frame at 10 seconds saved successfully!")
    else:
        print("Error: Could not read the frame.")

cap.release()