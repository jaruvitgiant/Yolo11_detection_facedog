from ultralytics import YOLO

model = YOLO('yolo11n.pt')

model.train(
    data="DogDatasets/data.yml",
    epochs=100,
    imgsz=224,
    batch=16,
    device="cpu",  
    workers=0,

    # การตั้งค่า Data Augmentation
    fliplr=0.5,                    # ความน่าจะเป็นในการพลิกรูปแนวนอน (0.5 = 50%)
    degrees=10.0,                     # หมุนภาพแบบสุ่ม ±10 องศา
    translate=0.1,                 # การเลื่อนตำแหน่งภาพแบบสุ่ม (10%)
    scale=0.2,                     # การย่อ/ขยายภาพแบบสุ่ม (±20%)
    shear=2.0,                     # การบิดภาพ (shear) แบบสุ่ม
    hsv_h=0.015,                   # ปรับค่า Hue ของสี (±0.015)
    hsv_s=0.7,                     # ปรับค่า Saturation ของสี (±70%)
    hsv_v=0.4,                     # ปรับค่า Value (ความสว่าง) (±40%)
    mosaic=0.7,                    # การทำ mosaic augmentation (รวมหลายภาพ) (ความน่าจะเป็น 70%)
    mixup=0.1,                     # การทำ mixup (ผสมภาพซ้อนกัน) (10%)
    
    # พารามิเตอร์เพิ่มเติม
    val=True,                         # ตั้งค่าให้รัน Validation ในระหว่างการฝึก
    save=True,                        # ตั้งค่าให้บันทึก checkpoint ของโมเดล
)


model.val()