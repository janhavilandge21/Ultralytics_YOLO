
# 🧠 Ultralytics YOLO Object Detection Project

This project implements YOLO (You Only Look Once) — a state-of-the-art, real-time object detection system — using the Ultralytics YOLOv8 library.
It can detect, track, and classify multiple objects in images or live video streams with high accuracy and speed.

# 📦 Features

✅ Real-time object detection using webcam or video files
✅ Support for multiple YOLO model variants (yolov8n, yolov8s, yolov8m, etc.)
✅ Custom dataset training capability
✅ Result visualization (bounding boxes, labels, and confidence scores)
✅ Easy integration with OpenCV and NumPy

# 🛠️ Installation

 Create and Activate Virtual Environment (optional)
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On macOS/Linux

 Install Dependencies
pip install ultralytics opencv-python numpy

# 🧩 How to Use
🔹 Run Detection on Image
from ultralytics import YOLO

# Load the model
model = YOLO("yolov8n.pt")

# Run inference on an image
results = model("sample.jpg")

# Display results
results.show()

🔹 Run Live Detection with Webcam
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    annotated = results[0].plot()
    cv2.imshow("YOLOv8 Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 🧠 Training on Custom Dataset

To train your model on your own dataset, create a YAML file specifying image paths and classes:

train: path/to/train/images
val: path/to/val/images

nc: 3
names: ['cat', 'dog', 'bird']


Then run:

yolo train model=yolov8n.pt data=custom.yaml epochs=100 imgsz=640

# 📊 Results

After running detection or training, results will be stored in the runs/ directory.
You can find:

Inference outputs (images/videos with boxes)

Training metrics (accuracy, mAP, loss graphs)


# 🤝 Contributing

1.Contributions are welcome!

2.Fork the repository

3.Create a new branch (feature/your-feature-name)

4.Commit your changes

5.Submit a pull request 🚀
