#!/usr/bin/env python3
"""
Convert classification dataset → YOLO detection format (3 classes).
Full-image bounding boxes (legitimate when disease spans the whole leaf).

Classes:
  0: healthy (sanatoasa)
  1: Fusarium oxysporum (fusarioza)
  2: Oidium belbahrii (fainare)
"""

import shutil
from pathlib import Path

SRC = Path("/Users/iustiniuliandodita/sera-hidroponica/basil_classification_dataset")
DST = Path("/Users/iustiniuliandodita/sera-hidroponica/basil_detection_3class")

CLASS_MAP = {
    "healthy":  0,
    "fusarium": 1,
    "powdery":  2,
}

SPLIT_MAP = {
    "train": "train",
    "val":   "val",
    "test":  "test",
}

def convert():
    if DST.exists():
        shutil.rmtree(DST)

    for split_src, split_dst in SPLIT_MAP.items():
        img_out = DST / "images" / split_dst
        lbl_out = DST / "labels" / split_dst
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for cls_name, cls_id in CLASS_MAP.items():
            src_dir = SRC / split_src / cls_name
            if not src_dir.exists():
                print(f"  SKIP {src_dir} (not found)")
                continue

            images = list(src_dir.iterdir())
            for img_path in images:
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue

                # Copy image
                dst_img = img_out / img_path.name
                shutil.copy2(img_path, dst_img)

                # Tight center box (0.85x0.85): avoids full-overlap NMS issues
                label_path = lbl_out / (img_path.stem + ".txt")
                label_path.write_text(f"{cls_id} 0.5 0.5 0.85 0.85\n")

            print(f"  {split_src}/{cls_name}: {len(images)} images → class {cls_id}")

    # Write data.yaml
    yaml_content = f"""path: {DST}
train: images/train
val: images/val
test: images/test

nc: 3
names:
  0: "healthy (sanatoasa)"
  1: "Fusarium oxysporum (fusarioza)"
  2: "Oidium belbahrii (fainare)"
"""
    (DST / "data.yaml").write_text(yaml_content)
    print(f"\ndata.yaml written to {DST / 'data.yaml'}")

    # Summary
    for split in ["train", "val", "test"]:
        n = len(list((DST / "images" / split).glob("*")))
        print(f"  {split}: {n} images")

if __name__ == "__main__":
    print("Building basil_detection_3class dataset...")
    convert()
    print("\nDone.")
