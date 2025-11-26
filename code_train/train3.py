from ultralytics import YOLO

model = YOLO('yolo11n.pt')

model.train(
    data="DogDatasets/data.yml",
    epochs=100,
    imgsz=224,
    batch=16,
    device="cpu",  
    workers=0,

    lr0=0.01,                       
    lrf=0.001,                        # Learning Rate สุดท้ายที่ต่ำลงเพื่อการ Fine-tune ที่แม่นยำขึ้น
    momentum=0.9,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,

    # การตั้งค่า Data Augmentation
    fliplr=0.5,                # ความน่าจะเป็นในการพลิกรูปแนวนอน (0.5 = 50%)
    degrees=15.0, # หมุนภาพแบบสุ่ม ±10 องศา
    translate=0.2, # การเลื่อนตำแหน่งภาพแบบสุ่ม (10%)
    scale=0.5, # การย่อ/ขยายภาพแบบสุ่ม (±20%)
    shear=2.0, # การบิดภาพ (shear) แบบสุ่ม

    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    mosaic=0.7,
    mixup=0.1,
    
    # การตั้งค่า Loss
    box=7.5,
    cls=0.5,
    dfl=1.5,
    label_smoothing=0.05,

    # พารามิเตอร์เพิ่มเติม
    val=True,                         # ตั้งค่าให้รัน Validation ในระหว่างการฝึก
    save=True,                        # ตั้งค่าให้บันทึก checkpoint ของโมเดล
)


model.val()