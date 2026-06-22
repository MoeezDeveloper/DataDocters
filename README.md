# 👁️ DiabEye — Diabetic Retinopathy Early Warning System

> An AI-powered clinical screening tool that detects Diabetic Retinopathy risk from retinal features using statistical analysis and machine learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Overview

**DiabEye** (branded as **DataDoctors**) is an interactive Streamlit web application built for the early detection of Diabetic Retinopathy (DR) — a leading cause of preventable blindness in diabetic patients.

The app combines rigorous statistical analysis with trained machine learning models to provide real-time, three-tier risk stratification: **Low**, **Medium**, and **High Risk** — designed with clinical sensitivity as a priority.

---

## 🚀 Features

| Module | Description |
|---|---|
| 🏠 **Home** | Dataset overview, key metrics, class distribution, sample data |
| 📊 **Visualization** | Histograms, scatter plots, correlation heatmap, boxplots |
| 📐 **Statistical Analysis** | Descriptive stats, confidence intervals, skewness analysis |
| 🎲 **Probability** | Prior/conditional probabilities, normal distribution fitting |
| 🧪 **Hypothesis Testing** | Independent t-tests, chi-square tests, p-value visualization |
| 🔍 **Outlier Detection** | IQR-based outlier detection with adjustable multiplier |
| 🤖 **Live Prediction** | Manual patient input OR random patient demo with real-time DR risk gauge |
| 📈 **Model Evaluation** | Confusion matrices, ROC curves, feature importance — side-by-side model comparison |

---

## 🧠 Machine Learning Models

Two classifiers are trained and benchmarked:

| Model | Accuracy | Recall | AUC-ROC |
|---|---|---|---|
| Logistic Regression | 70.11% | 92.74% | 0.8611 |
| **Random Forest ⭐** | **89.45%** | **99.17%** | **0.9844** |

**Key Design Decision:** A classification threshold of **0.30** (instead of the default 0.50) is used to maximize recall — because in medical screening, a missed diagnosis carries far greater cost than a false alarm.

---

## 📂 Project Structure

```
DiabEye/
│
├── app.py                          # Main Streamlit application
├── Retinopathy_Debrecen.csv        # Dataset (required)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 📊 Dataset

- **Source:** Retinopathy Debrecen Dataset — UCI Machine Learning Repository (SMOTE augmented version)
- **File Required:** `Retinopathy_Debrecen.csv` — place in the same directory as `app.py`
- **Patients:** 4,831 records
- **Features:** 18 retinal features + 1 target class
- **Class Balance:** 50.16% DR Positive | 49.84% DR Negative

### Feature Summary

| Feature | Type | Description |
|---|---|---|
| `quality` | Binary | Retinal image quality (0=poor, 1=acceptable) |
| `pre_screening` | Binary | Pre-screening result for severe abnormality |
| `ma1` – `ma6` | Continuous | Microaneurysm detections at 6 confidence levels |
| `exudate1` – `exudate8` | Continuous | Hard exudate measurements at various spatial levels |
| `macula_opticdisc_distance` | Continuous | Euclidean distance between macula and optic disc |
| `opticdisc_diameter` | Continuous | Normalized optic disc diameter |
| `am_fm_classification` | Binary | AM/FM-based classification result |
| `class` | Binary | **Target** — 0 = No DR, 1 = Has Diabetic Retinopathy |

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/DiabEye.git
cd DiabEye
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the Dataset

Place `Retinopathy_Debrecen.csv` in the root project directory.

### 4. Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📦 Requirements

```
streamlit
pandas
numpy
plotly
scipy
scikit-learn
```

Install all at once:

```bash
pip install streamlit pandas numpy plotly scipy scikit-learn
```

---

## 🔍 How It Works

1. **Data is loaded** from `Retinopathy_Debrecen.csv` and cached for performance
2. **Models are trained** at startup using an 80/20 train-test split with `StandardScaler` normalization
3. **Live Prediction** — input patient retinal values manually or pick random patients from the dataset
4. **Risk Gauge** — visual probability meter with color-coded zones (green / yellow / red)
5. **Risk Stratification:**
   - 🟢 **Low Risk** — P(DR) < 0.30
   - 🟡 **Medium Risk** — P(DR) 0.30 – 0.60
   - 🔴 **High Risk** — P(DR) > 0.60

---



## 📄 License

This project is for academic purposes under FAST NUCES. Dataset credit goes to the University of Debrecen, Hungary via the UCI Machine Learning Repository.
