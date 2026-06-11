#!/usr/bin/env python3
from pathlib import Path
import sys

root = Path("/Users/iustiniuliandodita/sera-hidroponica/basil_classification_dataset")
classes = ["healthy", "fusarium", "powdery"]
splits = ["train", "val", "test"]

print("=" * 60)
print("🔍 BASIL CLASSIFICATION DATASET VERIFICATION")
print("=" * 60)

if not root.exists():
    print(f"❌ Missing dataset: {root}")
    sys.exit(1)

ok = True
for split in splits:
    split_dir = root / split
    if not split_dir.exists():
        print(f"❌ Missing split: {split}")
        ok = False
        continue
    print(f"\n{split.upper()}")
    total = 0
    found_classes = [d.name for d in split_dir.iterdir() if d.is_dir()]
    if sorted(found_classes) != sorted(classes):
        print(f"❌ Class folders mismatch: {found_classes}")
        ok = False
    for cls in classes:
        n = len([f for f in (split_dir / cls).glob("*") if f.is_file()])
        total += n
        print(f"  {cls}: {n}")
        if n == 0:
            ok = False
    print(f"  TOTAL: {total}")

yaml_path = root / "data.yaml"
print(f"\nConfig: {'✅' if yaml_path.exists() else '❌'} {yaml_path}")

print("\n" + "=" * 60)
if ok:
    print("✅ DATASET OK FOR CLASSIFICATION TRAINING")
else:
    print("❌ DATASET HAS ISSUES")
    sys.exit(1)
