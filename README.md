![Sample Image](models_index/detect/train/val_batch0_pred.jpg)

Detect_dog_faceYOLO11 — คำอธิบายโฟลเดอร์

ภาพรวม
- โฟลเดอร์นี้รวมโค้ดและทรัพยากรสำหรับตรวจจับใบหน้า/ส่วนหัวสุนัขโดยใช้รุ่น YOLO (เวอร์ชันที่เก็บเป็น `yolo11n.pt`) และสคริปต์สำหรับการทดสอบกับชุดข้อมูล unseen

ไฟล์และโฟลเดอร์สำคัญ
- `yolo11n.pt`: ไฟล์โมเดลที่ฝึก (น้ำหนักของ YOLO รุ่นที่ใช้)
- `TestUnseen.py`: สคริปต์สำหรับรันการทดสอบบนภาพที่ไม่เคยเห็น (unseen) — ใช้สำหรับประเมินผลการตรวจจับ
- `pyproject.toml`: ข้อมูลโครงการ/การขึ้นต้นและพึ่งพา (ตรวจสอบ dependencies ที่จำเป็น)
- `code/`, `code_train/`, `codeTest/`: โค้ดตัวอย่างสำหรับการฝึกและทดสอบโมเดล
- `Augmentation_Train/`: สคริปต์และตัวอย่างสำหรับการทำ augmentation ขณะฝึก
### DogDatasets (External Link)
เพื่อไม่ให้ repository มีขนาดใหญ่ ชุดข้อมูลสุนัขจะไม่ถูกเก็บใน repo  
ดาวน์โหลดได้ที่นี่:[Download Dog Dataset](https://your-dataset-link.com)
- `runs/`: โฟลเดอร์ผลลัพธ์การฝึก/ทดลอง (logs, checkpoints)
- `cut_object/`: เครื่องมือ/สคริปต์สำหรับตัดวัตถุ (cropping) ที่เกี่ยวข้อง

acc 98%
