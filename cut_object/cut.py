from ultralytics import YOLO
import cv2
import os

class_names = ["FaceDog", "BodyDog"] 
model = YOLO("../runs/detect/train3/weights/best.pt")

base_dir = "crops0.8"
os.makedirs(base_dir, exist_ok=True)

results = model.predict(source="../DogsTest_image", conf=0.8)

for i, result in enumerate(results):
    img = result.orig_img  # ภาพต้นฉบับ (numpy array)
    filename = os.path.basename(result.path)      # เช่น dog1.jpg
    name, ext = os.path.splitext(filename) 

    for j, box in enumerate(result.boxes):
        # พิกัด bounding box (pixel)
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cls = int(box.cls[0])   # class index
        conf = float(box.conf[0])

        # กำหนดชื่อ class
        cls_name = class_names[cls] if cls < len(class_names) else f"class_{cls}"

        # สร้างโฟลเดอร์ class ถ้ายังไม่มี
        save_dir = os.path.join(base_dir, cls_name)
        os.makedirs(save_dir, exist_ok=True)

        # ครอป object
        crop = img[y1:y2, x1:x2]

        # เซฟเป็นไฟล์ใหม่
        save_path = os.path.join(save_dir, f"{name}_obj{j}_conf{conf:.2f}{ext}")
        cv2.imwrite(save_path, crop)

print("ครอปและแยกโฟลเดอร์ตาม class เสร็จแล้ว")
        