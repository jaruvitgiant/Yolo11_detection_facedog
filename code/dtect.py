from ultralytics import YOLO
model = YOLO("yolo11n.pt")

results = model("test/112.jpg")  # Predict on an image
results[0].show()