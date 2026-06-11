# Basil Disease Detection Dataset

YOLOv8 object detection dataset for basil plant disease identification.

## Classes (5 categories)
0. **healthy** - Healthy basil leaves
1. **downy_mildew** - Downy mildew fungal disease
2. **bacterial_leaf_spot** - Bacterial leaf spot disease
3. **leaf_miner_damage** - Damage from leaf-mining insects
4. **general_disease** - Other disease patterns

## Source
- Original: Roboflow (disease-detection-2r3sj/basil-health-detection)
- Link: https://universe.roboflow.com/disease-detection-2r3sj/basil-health-detection-9crhs

## Structure
```
basil_dataset/
├── images/
│   ├── train/  (1,167 images)
│   ├── val/    (334 images)
│   └── test/   (169 images)
├── labels/
│   ├── train/  (YOLO format annotations)
│   ├── val/
│   └── test/
├── data.yaml   (YOLOv8 configuration)
└── README.md   (this file)
```

## Training
```bash
python train_basil.py
```

## Inference
```bash
python infer_basil.py --image path/to/image.jpg
```
