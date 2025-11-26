from ultralytics import YOLO

# โหลดโมเดล YOLOv8n
model = YOLO("yolov8n.pt")

# กำหนดโฟลเดอร์ผลลัพธ์
project_folder = r"C:\projectYOLO11\my_results"  # โฟลเดอร์หลัก
run_name = "train_experiment1"                    # ชื่อ sub-folder สำหรับ run นี้

train_results = model.train(
    data="DataSampling/datasets/dataTEst.yml",  # path dataset config
    epochs=10,
    imgsz=640,
    device="cpu",
    workers=0,
    project=project_folder,  # บันทึกผลลัพธ์ที่นี่
    name=run_name,           # ชื่อโฟลเดอร์ย่อยสำหรับ run นี้
    exist_ok=True            # ถ้าโฟลเดอร์นี้มีอยู่แล้ว จะเขียนทับ
)

# ตรวจสอบ metrics หลัง train
metrics = model.val()