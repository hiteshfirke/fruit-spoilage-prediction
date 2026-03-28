# 🍌 Fruit Spoilage Prediction System

> A research-grade computer vision system that predicts banana ripeness stages and estimates shelf life using MobileNetV2 + environmental fusion.

---

## 📌 Project Overview

This project is a full-stack AI web application that classifies bananas into **4 ripeness stages** from a photo and estimates **days remaining before spoilage** by combining the vision model output with ambient **temperature** and **humidity** readings.

Built as part of the **EDI Project at VIT Pune**, this system demonstrates how transfer learning and environmental modelling can be combined for real-world food quality monitoring.

---

## 🎯 What It Does

| Input | Output |
|---|---|
| 📸 Banana image | 🏷️ Ripeness stage (one of 4) |
| 🌡️ Temperature (°C) | 📅 Estimated days before spoilage |
| 💧 Humidity (%) | 📊 Model confidence % |

### Ripeness Stages Detected
| Stage | Label | Description |
|---|---|---|
| 0 | `stage0_unripe` | Green, not ready |
| 1 | `stage1_ripe` | Yellow, ready to eat |
| 2 | `stage2_overripe` | Spotted, use soon |
| 3 | `stage3_rotten` | Black, spoiled |

---

## 🧠 Model Architecture

- **Backbone:** MobileNetV2 (pre-trained on ImageNet, frozen)
- **Head:** GlobalAveragePooling2D → Dropout(0.3) → Dense(128, ReLU) → Dense(4, Softmax)
- **Optimizer:** Adam (lr=1e-4)
- **Loss:** Sparse Categorical Crossentropy with class weights
- **Input Size:** 224 × 224 × 3

### Augmentation Pipeline
Random flip · Random rotation · Random zoom · Random contrast · Random brightness

### Environmental Fusion
The predicted ripeness stage is passed into a **biological spoilage model** that uses temperature and humidity to compute estimated days remaining, based on empirical banana ripening behaviour.

---

## 📊 Results

| Metric | Value |
|---|---|
| ✅ Test Accuracy | **92.6%** |
| 🖼️ Training Images | 10,000+ |
| 🏷️ Classes | 4 |
| 📦 Export Formats | `.keras` + `.tflite` |

---

## 🗂️ Project Structure

```
fruit-spoilage-prediction/
│
├── train_mobilenet.py        # ML training script (MobileNetV2)
│
├── app.py                    # Flask backend server
│
├── templates/
│   ├── index.html            # Landing / hero page
│   ├── main.html             # Prediction UI
│   └── architecture.html     # Research & methodology page
│
├── static/
│   ├── index.css
│   ├── main.css
│   ├── architecture.css
│   └── fruit_store_bg.mp4    # Background video
│
├── dataset/                  # ⚠️ Not included — see below
│   ├── train/
│   └── test/
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/hiteshfirke/fruit-spoilage-prediction.git
cd fruit-spoilage-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
The dataset (~3GB of banana images) is not included in this repo due to size.

📦 **Download here:** [Google Drive / Kaggle Link — add your link here]

Place it in the following structure:
```
dataset/
├── train/
│   ├── stage0_unripe/
│   ├── stage1_ripe/
│   ├── stage2_overripe/
│   └── stage3_rotten/
└── test/
    ├── stage0_unripe/
    ├── stage1_ripe/
    ├── stage2_overripe/
    └── stage3_rotten/
```

### 4. Train the Model
```bash
python train_mobilenet.py
```
This will save `banana_stage_mobilenet_final.keras` and `banana_stage_mobilenet_final.tflite` inside the `dataset/` folder.

### 5. Run the Web App
```bash
python app.py
```
Then open your browser and go to: `http://localhost:5000`

---

## 🖥️ Web App Pages

| Page | Route | Description |
|---|---|---|
| Landing Page | `/` | Hero intro with background video |
| Predictor UI | `/main` | Upload image + enter temp/humidity |
| Architecture | `/architecture` | Research methodology for reviewers |

---

## 📦 Requirements

Create a `requirements.txt` with:
```
tensorflow>=2.12
flask
matplotlib
numpy
pillow
```

---

## 👥 Team

| Name | Role | GitHub |
|---|---|---|
| **Hitesh Firke** | 🤖 ML Model & Training (MobileNetV2, TFLite export) | [@hiteshfirke](https://github.com/hiteshfirke) |
| **Kapil Hire** | 🗃️ Data Processing & Cleaning | [@kapilhire9b30-spec](https://github.com/kapilhire9b30-spec) |
| **Harsh Dhavane** | ⚙️ Backend Development (Flask API) | — |
| **Varad Hivarkar** | 🎨 Frontend Development (Web UI) | [@varadhivarkar1703-create](https://github.com/varadhivarkar1703-create) |

> 📍 EDI Project · VIT Pune

---

## 📄 License

This project is for academic/research purposes as part of VIT Pune's EDI programme.

---

## 🙏 Acknowledgements

- [MobileNetV2](https://arxiv.org/abs/1801.04381) — Google Brain
- TensorFlow / Keras
- VIT Pune — EDI Programme
