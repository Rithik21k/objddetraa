import numpy as np
import cv2
from ultralytics import YOLO
import random

# Load COCO class names from a file
with open("utils/coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Generate random colors for each class
detection_colors = []
for _ in range(len(class_list)):
    detection_colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

# Load YOLOv8 model (make sure the path to your model file is correct)
model = YOLO("yolov8n.pt")  # YOLOv8n is a smaller version for faster performance

# Set frame width and height
frame_wid = 640
frame_hyt = 480

# Open video file or capture device (e.g., webcam)
cap = cv2.VideoCapture("inference/videos/afriq0.MP4")  # Use video file or webcam (change to 0 for webcam)

if not cap.isOpened():
    print("Cannot open camera or video file")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Resize frame if necessary (optional)
    frame = cv2.resize(frame, (frame_wid, frame_hyt))

    # Perform object detection
    results = model.predict(source=frame, conf=0.45, save=False)

    # Process detection results
    for result in results:
        boxes = result.boxes  # Detected bounding boxes

        for box in boxes:
            clsID = int(box.cls.numpy()[0])  # Class ID of the object
            conf = float(box.conf.numpy()[0])  # Confidence of the detection
            bb = box.xyxy.numpy()[0]  # Bounding box coordinates

            # Draw the bounding box on the frame
            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[clsID],
                3
            )

            # Display class name and confidence score
            cv2.putText(
                frame,
                f"{class_list[clsID]} {conf:.2f}%",
                (int(bb[0]), int(bb[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

    # Show the frame with detection results
    cv2.imshow("ObjectDetection", frame)

    # Exit the loop when "Q" is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object and close windows
cap.release()
cv2.destroyAllWindows()
