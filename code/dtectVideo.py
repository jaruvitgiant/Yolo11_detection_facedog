from ultralytics import YOLO

# โหลดโมเดลที่ train เสร็จ
model = YOLO("best.pt")

# รันบนวิดีโอ และบันทึกผลลัพธ์
results = model.predict(
    source="test/dogtest.mp4",   # path ของวิดีโอ
    save=True,                   # บันทึกไฟล์ผลลัพธ์
    show=True,                   # แสดงผลทีละเฟรม (ต้องมี GUI)
    conf=0.5                     # ค่า confidence threshold
)

print("เสร็จแล้ว! ผลลัพธ์จะถูกบันทึกที่ runs/detect/predict/")
