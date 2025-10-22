from ultralytics import YOLO

# Load the YOLO12 model
model = YOLO("yolo12n.pt")  # adjust path/name as needed

# Run prediction
results = model.predict(
    source=r"C:\Users\JANHAVI\Desktop\Yolo.jpg",
    conf=0.5
)

print(results)
