# 🌿 Seră Hidroponică Smart - YOLOv8 (busuioc/mentă)

Repo-ul a fost curățat de tot ce ține de training/dataset pentru tomate. Pipeline-ul curent folosește un dataset iNaturalist pentru **busuioc** și **mentă** (imagini + bbox pe plantă), util pentru „detecție prezență plantă” / demo.

De asta NU vezi clase gen `early_blight`: iNaturalist nu îți dă etichete de boală. Ca să ai clasificare/Detecție pe boli/deficiențe trebuie:
1) un `data.yaml` cu **clasele de boală**
2) label-uri YOLO care folosesc acele ID-uri (bbox pe simptome).

Am adăugat un template:
- `examples/disease_data_4cls.yaml`
- `init_disease_dataset.py` (îți creează structura + `datasets/herbs_disease/data.yaml`)

## 🚀 Quick start (Windows)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 1) Descarcă dataset iNaturalist (local, ignorat din git)
python download_inat_yolo.py --out inat_datasets\inat_basil_mint_20k --per-class 10000 --val-per-class 1000 --workers 16

# 2) Train
python train.py --data inat_datasets\inat_basil_mint_20k\data.yaml --epochs 100 --model n --name inat_basil_mint_20k_y8n
```

Rezultatele se salvează în `runs\detect\<name>\` (weights + plots generate de Ultralytics).

## 📁 Scripturi utile
- `download_inat_yolo.py` – descarcă imagini din iNaturalist și construiește dataset YOLO.
- `train.py` – antrenament YOLOv8 (CUDA/MPS/CPU) cu auto-batch + cache auto.
- `setup_check.py` – verifică rapid Python/PyTorch/Ultralytics + dacă există dataset local.
