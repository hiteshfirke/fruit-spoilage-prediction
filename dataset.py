import os
import matplotlib.pyplot as plt

# CHANGE THIS PATH to your training dataset folder
DATASET_DIR = "dataset/train"

classes = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])

counts = []

for cls in classes:
    cls_path = os.path.join(DATASET_DIR, cls)
    num_images = len([
        f for f in os.listdir(cls_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])
    counts.append(num_images)

plt.figure()
plt.bar(classes, counts)
plt.xlabel("Ripeness Stage")
plt.ylabel("Number of Images")
plt.title("Dataset Class Distribution")

plt.tight_layout()
plt.savefig("dataset_distribution.png", dpi=300)
plt.show()
