# import os
# from ultralytics import YOLO

# # โหลดโมเดลที่เทรนเสร็จแล้ว (แก้ path ถ้าไฟล์อยู่ที่อื่น)
# model = YOLO("runsYolo11/detect/train/weights/best.pt")

# # โฟลเดอร์ภาพทดสอบ
# test_folder = "DogsTest_image"

# # สร้าง output folder
# save_dir = "runs/detect/testDog_results"
# os.makedirs(save_dir, exist_ok=True)

# # รัน predict
# results = model.predict(
#     source=test_folder,    # โฟลเดอร์รูป
#     save=True,             # เซฟภาพผลลัพธ์
#     project="runs/detect", # โฟลเดอร์หลัก
#     name="testDog_results",# subfolder
#     conf=0.25              # กำหนด threshold (0.25 = ตัดกรอบที่มั่นใจต่ำกว่าออก)
# )

# print(f"เสร็จแล้ว! ผลลัพธ์ถูกบันทึกไว้ที่: {save_dir}")


import os
import csv
from ultralytics import YOLO

# โหลดโมเดลที่เทรนเสร็จแล้ว
model = YOLO("runsYolo11_Train/detect/train/weights/best.pt")

# โฟลเดอร์ภาพทดสอบ
test_folder = "DogsTest_image"

save_dir = "runsYolo11_Test/detect/testDog_results"
os.makedirs(save_dir, exist_ok=True)

# รัน predict และเก็บผลลัพธ์
results = model.predict(
    source=test_folder,
    save=True,
    project="runsYolo11_Test/detect",
    name="testDog_results",
    conf=0.25
)

# ไฟล์ CSV สำหรับเก็บผลลัพธ์
csv_file = os.path.join(save_dir, "detection_results.csv")

with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "class", "confidence", "x1", "y1", "x2", "y2"])  # header

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])  # class index
            conf = float(box.conf[0])  # confidence
            x1, y1, x2, y2 = map(float, box.xyxy[0])  # bounding box
            writer.writerow([
                os.path.basename(result.path),
                model.names[cls],
                f"{conf:.4f}",
                f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}"
            ])

print(f" เสร็จแล้ว! ผลลัพธ์ภาพอยู่ที่: {save_dir}")
print(f" ไฟล์ CSV เก็บผลลัพธ์ที่: {csv_file}")

