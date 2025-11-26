from ultralytics import YOLO

model = YOLO("runs/detect/train3/weights/best.pt")

# ประเมินผล test แล้วเก็บผลไว้ในโฟลเดอร์ที่กำหนด
metrics = model.val(
    data="DogDatasets/data.yml",
    split="test",
    project="TestUnseen",
    name="dog_test"
)
