from flask import Flask, request, redirect, url_for, render_template
import os
import cv2
from ultralytics import YOLO
import random
import numpy as np

app = Flask(__name__)

# Directory where uploaded videos will be stored
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load class list and generate colors (same as before)
my_file = open("utils/coco.txt", "r")
class_list = my_file.read().split("\n")
my_file.close()

detection_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(class_list))]

# Load YOLO model
model = YOLO("weights/yolov8n.pt", "v8")

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return 'No file part'
    
    file = request.files['video']
    
    if file.filename == '':
        return 'No selected file'
    
    if file:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(video_path)

        # Call the object detection function here
        detect_objects(video_path)
        
        return 'Video uploaded and processed successfully!'

def detect_objects(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Cannot open video file")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detect_params = model.predict(source=[frame], conf=0.45, save=False)
        DP = detect_params[0].numpy()

        if len(DP) != 0:
            for i in range(len(detect_params[0])):
                boxes = detect_params[0].boxes
                box = boxes[i]
                clsID = box.cls.numpy()[0]
                conf = box.conf.numpy()[0]
                bb = box.xyxy.numpy()[0]

                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    detection_colors[int(clsID)],
                    3
                )

                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(
                    frame,
                    class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
                    (int(bb[0]), int(bb[1]) - 10),
                    font,
                    1,
                    (255, 255, 255),
                    2
                )

        cv2.imshow("ObjectDetection", frame)
        
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(debug=True)
