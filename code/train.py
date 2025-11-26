from ultralytics import YOLO
import torch_directml
model = YOLO("yolov8n.pt")

results = model.train(
    data="DogDatasets/data.yml",  
    # batch = 8,
    epochs=100, 
    imgsz=640,  
    device=0,  
    workers=0,  # Number of data loading workers
)
metrics = model.val()



# from ultralytics import YOLO
# model = YOLO("yolov8n.pt")
# train_results = model.train(
#     data="DataSampling/datasets/dataTEst.yml",  # Path to dataset configuration file
#     epochs=50,
#     # batch = 8,
#     imgsz=416,  # Image size for training
#     device="cpu",
#     save=True,           # บันทึก weights ทุก epoch
#     save_period=1,       # เก็บ weights ทุก 1 epoch
#     plots=True,          # บันทึกกราฟผลการเทรน
#     verbose=True,
#     workers = 0  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
# )
# metrics = model.val()

