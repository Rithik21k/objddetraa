# Load YOLOv8n-cls, train it on mnist160 for 3 epochs and predict an image with it
from ultralytics import YOLO

model = YOLO('weights/yolov8n-cls.pt')  # load a pretrained YOLOv8n classification model
model.train(data='C:\\Users\\rithi\\OneDrive\\Desktop\\Documents\\yolov8-silva-main[1]\\yolov8-silva-main\\datasets\\animals', epochs=100)  # train the model
model('inference/images/bird.jpeg')  # predict on an image