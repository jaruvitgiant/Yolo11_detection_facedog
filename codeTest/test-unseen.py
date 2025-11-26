# from ultralytics import YOLO

# model = YOLO("runs/detect/train/weights/best.pt")

# results = model.predict(
#     source="DogDatasets/test/images",  # โฟลเดอร์ภาพ test
#     imgsz=224,
#     conf=0.25,         # confidence threshold
#     save=True,         # บันทึกภาพพร้อม bounding box
#     save_txt=True,     # บันทึก label prediction เป็น .txt (YOLO format)
    
# )


from ultralytics import YOLO
import os
model = YOLO("../runs/detect/train3/weights/best.pt")

results = model.predict(
    source="../image1.jpg",
    imgsz=224,
    conf=0.25,     
    save=True,      
    save_txt=True,   # บันทึก label prediction (YOLO format)
)

# สร้างโฟลเดอร์สำหรับเก็บผลลัพธ์เพิ่มเติม
os.makedirs("results_conf", exist_ok=True)

# loop เก็บค่า confidence
for i, result in enumerate(results):
    # result.boxes.xyxy -> ค่าพิกัด [x1,y1,x2,y2]
    # result.boxes.conf -> confidence
    # result.boxes.cls -> class index
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    save_path = os.path.join("results_conf", f"image_{i}.txt")

    with open(save_path, "w") as f:
        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = box
            f.write(f"{int(cls)} {conf:.4f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")

print("บันทึกผลเรียบร้อยแล้ว ที่โฟลเดอร์ results_conf/")
