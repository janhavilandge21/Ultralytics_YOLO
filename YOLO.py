from ultralytics import YOLO
import numpy

model = YOLO("yolov8n.pt", "v8")

detection_output = model.predict(source=r"C:\Users\JANHAVI\Desktop\Yolo.jpg", conf=0.25, save = True)

print(detection_output)