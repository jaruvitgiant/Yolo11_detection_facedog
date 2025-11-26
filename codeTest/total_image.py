import os
from PIL import Image
import math

# โฟลเดอร์ที่เก็บภาพ predict ที่ YOLO สร้างไว้
img_dir = "runsYolo11_Test/detect/testDog_results2"

# กำหนดไฟล์ output
output_file = os.path.join(img_dir, "combined_results.jpg")

# ดึงไฟล์ภาพทั้งหมด (เอาเฉพาะ .jpg, .png)
images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]

if not images:
    raise FileNotFoundError("ไม่พบไฟล์ภาพในโฟลเดอร์ predict")

# โหลดภาพทั้งหมด
loaded_images = [Image.open(img) for img in images]

# ขนาดของแต่ละภาพ (resize ให้เท่ากันก่อน)
width, height = loaded_images[0].size
loaded_images = [img.resize((width, height)) for img in loaded_images]

# กำหนด grid layout (เช่น 3 คอลัมน์)
cols = 3
rows = math.ceil(len(loaded_images) / cols)

# สร้าง canvas ว่าง
combined = Image.new("RGB", (cols * width, rows * height), color=(255, 255, 255))

# วางภาพลงไปใน grid
for idx, img in enumerate(loaded_images):
    x = (idx % cols) * width
    y = (idx // cols) * height
    combined.paste(img, (x, y))

# เซฟไฟล์รวม
combined.save(output_file, quality=95)

print(f"✅ รวมภาพเสร็จแล้ว! บันทึกไว้ที่: {output_file}")
