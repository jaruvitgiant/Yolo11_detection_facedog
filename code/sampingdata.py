import os
import random
import shutil

source_folder = "TrainDog/datasets/labels/val"     # โฟลเดอร์ต้นทางที่มีรูปทั้งหมด
dest_folder = "DataSampling/datasets/valid/labels"   # โฟลเดอร์ปลายทางสำหรับเก็บรูปที่สุ่ม
sample_size = 30                            # จำนวนรูปที่ต้องการสุ่ม
# -------------------------------

# สร้างโฟลเดอร์ปลายทางถ้ายังไม่มี
os.makedirs(dest_folder, exist_ok=True)

# ดึงรายชื่อไฟล์รูปทั้งหมด
all_files = [f for f in os.listdir(source_folder) 
             if f.lower().endswith(('.txt', '.jpg'))]
print(f"พบรูปทั้งหมด {len(all_files)} ไฟล์")

# เช็คว่ามีไฟล์มากพอสำหรับสุ่มหรือไม่
# if sample_size > len(all_files):
#     sample_size = len(all_files)
#     print(f"⚠️ ไฟล์น้อยกว่าจำนวนที่ต้องการสุ่ม จะสุ่ม {sample_size} ไฟล์แทน")

# สุ่มไฟล์
sampled_files = random.sample(all_files, sample_size)

for file_name in sampled_files:
    src_path = os.path.join(source_folder, file_name)
    dst_path = os.path.join(dest_folder, file_name)
    shutil.copy2(src_path, dst_path)

print(f"สุ่มและคัดลอกไฟล์แล้ว {len(sampled_files)} ไฟล์ ไปที่ {dest_folder}")
