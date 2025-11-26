import pandas as pd
import matplotlib.pyplot as plt

# โหลดไฟล์ผลลัพธ์
df = pd.read_csv("my_results/train_experiment1/results.csv")

print(df.head())  # ดูข้อมูล 5 แถวแรก

# วาดกราฟ acc (mAP50) และ loss
plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
plt.plot(df['epoch'], df['train/box_loss'], label='Box Loss')
plt.plot(df['epoch'], df['train/cls_loss'], label='Cls Loss')
plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.title("Training Results")
plt.show()
