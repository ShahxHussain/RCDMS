from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Using YOLOv8 Nano model

# Train the model on the custom dataset 
model.train(
    data="E:/FYP Data/60/Car-Parts-Detection-2/data.yaml",  # Change to your dataset path
    epochs=5,
    imgsz=640,
    batch=4,
    workers=2,
    device="cpu"
)
