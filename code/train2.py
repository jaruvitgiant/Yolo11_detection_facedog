# from ultralytics import YOLO
# import torch_directml

# # dml = torch_directml.device()
# # print("Detected GPU:", dml)
# model = YOLO("yolov8n.pt")
# results = model.train(
#     data="DogDatasets/data.yml",  
#     # batch = 8,
#     epochs=50, 
#     imgsz=640,  
#     device='dml',  
#     workers=0,  # Number of data loading workers
# )
# metrics = model.val()



from ultralytics import YOLO
import torch_directml

# สร้าง DirectML device
dml = torch_directml.device()
print("Detected GPU:", dml)

# โหลดโมเดล
model = YOLO("yolov8n.pt")

# เทรน YOLO บน AMD GPU ผ่าน DirectML
model.train(
    data="DogDatasets/data.yml",
    epochs=50,
    device=dml,
    half=False,
    compile=False,
    amp=False
)

