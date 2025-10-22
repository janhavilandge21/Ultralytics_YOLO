import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO(r"C:\Users\JANHAVI\Desktop\CNN\yolov8n.pt")

# Initialize the video capture object
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Create tracker (KCF)
tracker = cv2.legacy.TrackerKCF_create()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform inference
    results = model.predict(frame)

    # Annotate frame
    annotated_frame = results[0].plot()

    # Display
    cv2.imshow('YOLOv8 Tracking', annotated_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
