import pandas as pd
import matplotlib.pyplot as plt
import os

# Path ไฟล์ CSV ที่ได้จากการ detect
csv_file = "runsYolo11_Test/detect/testDog_results/detection_results.csv"
save_plot = "runsYolo11_Test/detect/testDog_results/results_summary.png"

# โหลดข้อมูล
df = pd.read_csv(csv_file)

# นับจำนวน object ต่อ class
counts = df["class"].value_counts()

# สร้างกราฟแท่ง
plt.figure(figsize=(8, 6))
counts.plot(kind="bar")
plt.title("Detection Results per Class")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()

# เซฟเป็นรูป
plt.savefig(save_plot, dpi=300)
plt.close()

print(f"กราฟถูกบันทึกที่: {save_plot}")
