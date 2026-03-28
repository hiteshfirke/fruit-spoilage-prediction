import os
from collections import Counter

# DATASET_ROOT relative to this script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, "dataset")

for split in ["train", "test"]:
    split_dir = os.path.join(DATASET_ROOT, split)
    if not os.path.isdir(split_dir):
        print(f"[WARN] Split folder not found: {split_dir}")
        continue

    print(f"\n=== {split.upper()} ===")
    counts = Counter()

    for cls in os.listdir(split_dir):
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue

        n = sum(
            1 for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        counts[cls] = n

    for cls, n in sorted(counts.items()):
        print(f"{cls:20s}: {n} images")
