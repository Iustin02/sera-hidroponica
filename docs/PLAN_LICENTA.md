# 🌱 Seră Hidroponică Smart - Plan Licență

## 📊 Status Proiect: 35% Complet

---

## ✅ CE AM REALIZAT (2 Februarie 2026)

### 1. Infrastructură ML Setup ✅
- [x] Mediu Python virtual configurat
- [x] PyTorch 2.8.0 cu MPS (Apple Silicon)
- [x] Ultralytics YOLOv8 instalat
- [x] OpenCV, Pandas, Matplotlib configurate

### 2. Model YOLOv8 Antrenat ✅
- [x] **Model:** YOLOv8 Nano (3M parametri)
- [x] **Dataset:** 180 imagini sintetice (6 clase)
- [x] **Training:** 100 epoci completate
- [x] **Performanță:** mAP@50 = 81.5%
- [x] **Timp antrenare:** ~59 minute pe M1 Pro

### 3. Clase Detectate ✅
| Clasă | Descriere |
|-------|-----------|
| healthy | Frunze sănătoase |
| bacterial_spot | Pete bacteriene |
| early_blight | Mană timpurie |
| late_blight | Mană târzie |
| leaf_mold | Mucegai pe frunze |
| septoria_leaf_spot | Pete Septoria |

### 4. Cod Cross-Platform ✅
- [x] Detecție automată GPU (CUDA/MPS/CPU)
- [x] Script `train.py` hibrid
- [x] Compatibil Mac M1 Pro + PC RTX 3060

### 5. Repository GitHub ✅
- [x] **URL:** https://github.com/Iustin02/sera-hidroponica
- [x] Repository privat creat
- [x] Alias `pushsera` pentru push rapid

---

## 🎯 CE TREBUIE SĂ FACEM

### PRIORITATE ÎNALTĂ 🔴

#### 1. Dataset Real (Nu sintetic!)
- [ ] Descărcare dataset PlantVillage (54,000+ imagini)
- [ ] Sau fotografiere manuală plante Micro Tom
- [ ] Augmentare date (rotații, flip, brightness)
- [ ] Split: 70% train, 20% val, 10% test

#### 2. Re-antrenare cu Date Reale
- [ ] Training pe PC cu RTX 3060 (mai rapid)
- [ ] 200-300 epoci pentru convergență
- [ ] Fine-tuning hiperparametri
- [ ] Target: mAP@50 > 90%

#### 3. Hardware Seră
- [ ] Raspberry Pi 4 sau Jetson Nano
- [ ] Cameră USB/CSI pentru capturare
- [ ] Senzori: temperatură, umiditate, pH, EC
- [ ] Pompe și valve pentru hidroponică

### PRIORITATE MEDIE 🟡

#### 4. Interfață Web/Mobile
- [ ] Dashboard Flask/FastAPI
- [ ] Vizualizare live cameră
- [ ] Grafice senzori în timp real
- [ ] Alertă când detectează boală

#### 5. Automatizare Seră
- [ ] Control automat iluminare
- [ ] Dozare nutrienți automată
- [ ] Ventilație bazată pe temperatură
- [ ] Irigare programată

### PRIORITATE SCĂZUTĂ 🟢

#### 6. Documentație Licență
- [ ] Introducere și motivație
- [ ] State of the art (related work)
- [ ] Metodologie
- [ ] Rezultate și discuții
- [ ] Concluzii

---

## 📅 PROGRAM 5 ZILE (3-7 Februarie 2026)

### 📌 Ziua 1 - Luni 3 Februarie
**Focus: Dataset Real**
| Oră | Task | Durată |
|-----|------|--------|
| 09:00-11:00 | Descărcare PlantVillage dataset | 2h |
| 11:00-13:00 | Organizare și curățare date | 2h |
| 14:00-16:00 | Creare script augmentare | 2h |
| 16:00-18:00 | Pregătire structură YOLO | 2h |

**Deliverable:** Dataset organizat cu 10,000+ imagini

---

### 📌 Ziua 2 - Marți 4 Februarie
**Focus: Training Model Real**
| Oră | Task | Durată |
|-----|------|--------|
| 09:00-10:00 | Setup PC RTX 3060 (clone repo) | 1h |
| 10:00-12:00 | Verificare CUDA și test | 2h |
| 12:00-18:00 | Training 200 epoci (background) | 6h |
| 18:00-19:00 | Analiză rezultate preliminare | 1h |

**Deliverable:** Model antrenat pe date reale, mAP > 85%

---

### 📌 Ziua 3 - Miercuri 5 Februarie
**Focus: Hardware Setup**
| Oră | Task | Durată |
|-----|------|--------|
| 09:00-11:00 | Setup Raspberry Pi / Jetson | 2h |
| 11:00-13:00 | Instalare PyTorch + YOLOv8 | 2h |
| 14:00-16:00 | Test cameră și capturare | 2h |
| 16:00-18:00 | Optimizare inferență (TensorRT) | 2h |

**Deliverable:** Detecție funcțională pe edge device

---

### 📌 Ziua 4 - Joi 6 Februarie
**Focus: Interfață Web**
| Oră | Task | Durată |
|-----|------|--------|
| 09:00-11:00 | Setup Flask/FastAPI | 2h |
| 11:00-13:00 | Endpoint detecție imagini | 2h |
| 14:00-16:00 | Frontend simplu (HTML/JS) | 2h |
| 16:00-18:00 | Live video stream | 2h |

**Deliverable:** Dashboard web funcțional

---

### 📌 Ziua 5 - Vineri 7 Februarie
**Focus: Integrare & Documentare**
| Oră | Task | Durată |
|-----|------|--------|
| 09:00-11:00 | Integrare completă sistem | 2h |
| 11:00-13:00 | Testare end-to-end | 2h |
| 14:00-16:00 | Fix bugs și optimizări | 2h |
| 16:00-18:00 | Documentare cod și README | 2h |

**Deliverable:** MVP funcțional pentru demo

---

## 🛠️ RESURSE NECESARE

### Hardware
- [x] MacBook Pro M1 Pro (development)
- [x] PC RTX 3060 12GB (training)
- [ ] Raspberry Pi 4 / Jetson Nano
- [ ] Cameră USB HD
- [ ] Senzori IoT

### Software
- [x] Python 3.9+
- [x] PyTorch + Ultralytics
- [ ] Flask/FastAPI
- [ ] TensorRT (optimizare)
- [ ] MQTT (IoT)

### Datasets
- [ ] PlantVillage (Kaggle)
- [ ] Plant Pathology 2020
- [ ] Fotografii proprii

---

## 📈 METRICI SUCCES

| Metric | Actual | Target |
|--------|--------|--------|
| mAP@50 | 81.5% | >90% |
| FPS (Jetson) | - | >15 |
| Dataset size | 180 | >5000 |
| Latență detecție | - | <200ms |

---

## 📞 CONTACTE UTILE

- **Coordonator licență:** [Nume profesor]
- **GitHub:** https://github.com/Iustin02/sera-hidroponica
- **Documentație:** [Link Google Docs]

---

*Ultima actualizare: 2 Februarie 2026*
