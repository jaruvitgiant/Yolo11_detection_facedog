import os
import shutil
images_folder = "DogDatasets/valid/images"   # โฟลเดอร์ images ที่ต้องการเช็ค
source_folder = "DogTrain/datasets/labels/val"                                # โฟลเดอร์ต้นทางที่มีไฟล์ทั้งหมด
dest_folder = "DogDatasets/valid/labels" # โฟลเดอร์ปลายทางที่จะเก็บไฟล์ที่เจอ

os.makedirs(dest_folder, exist_ok=True)
# อ่านชื่อไฟล์ทั้งหมดในโฟลเดอร์ images (ไม่เอานามสกุล)
image_files = [os.path.splitext(f)[0] for f in os.listdir(images_folder)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"มีไฟล์ใน images/ {len(image_files)} ไฟล์")

# loop หาว่าใน source มีไฟล์ที่ตรงกับชื่อใน images ไหม
count = 0
for base_name in image_files:
    for ext in ('.txt','.jpg', '.jpeg', '.png'):
        src_file = os.path.join(source_folder, base_name + ext)
        if os.path.exists(src_file):
            shutil.copy2(src_file, dest_folder)   # หรือใช้ shutil.move ถ้าต้องการย้าย
            count += 1
            break

print(f"✅ คัดลอกไฟล์สำเร็จ {count} ไฟล์ ไปยัง {dest_folder}")
