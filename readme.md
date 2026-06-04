# 🛡️ UPI Fraud Detector

Real-time fraud detection system built using Machine Learning and Deep Learning.

## 🎯 Project Overview
This project detects fraudulent transactions using a multi-model approach combining unsupervised and supervised learning techniques.

## 🛠️ Tech Stack
| Component | Technology |
|-----------|------------|
| Language | Python 3.10 |
| ML Models | Scikit-learn |
| Deep Learning | PyTorch |
| UI Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

## 🔄 Project Flow
Raw Transaction Data
↓

Preprocessing + SMOTE (Fix class imbalance)
↓
K-Means Clustering (Normal vs Suspicious groups)
↓
Random Forest Classifier (Fraud/Not Fraud)
↓
PyTorch Neural Network (Fraud Risk Score 0-100)
↓
Streamlit Dashboard (Live predictions)

## 📊 Dataset
- **Source:** Kaggle — Credit Card Fraud Detection (ULB)
- **Size:** 284,807 transactions
- **Fraud cases:** 492 (0.17%)
- **Features:** 30 (V1-V28 PCA transformed + Amount + Time)

## 📈 Model Performance

| Model | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Random Forest | 82% | 82% | 82% |
| PyTorch Neural Network | 58% | 86% | 69% |

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/DhairyaJani2014/upi-fraud-detector.git
cd upi-fraud-detector
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download dataset
- Go to [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Download `creditcard.csv`
- Place it in `data/raw/` folder

### 5. Run notebooks in order
notebooks/01_EDA.ipynb
notebooks/02_Preprocessing.ipynb
notebooks/03_ML_Models.ipynb
notebooks/04_PyTorch_NN.ipynb

### 6. Launch Streamlit app
```bash
streamlit run app/app.py
```

## 📁 Project Structure
upi-fraud-detector/
├── data/
│   ├── raw/              → Original dataset
│   └── processed/        → Cleaned data
├── notebooks/
│   ├── 01_EDA.ipynb      → Data exploration
│   ├── 02_Preprocessing.ipynb → Data cleaning + SMOTE
│   ├── 03_ML_Models.ipynb    → K-Means + Random Forest
│   └── 04_PyTorch_NN.ipynb   → Neural Network
├── app/
│   └── app.py            → Streamlit dashboard
├── models/               → Saved trained models
├── requirements.txt
└── README.md

## 🎯 Resume Description
UPI Transaction Fraud Detection System

Built end-to-end fraud detection pipeline using K-Means
clustering and Random Forest classifier (82% recall)
Designed PyTorch Neural Network with Autograd for
real-time fraud risk scoring (86% recall, 0-100 score)
Deployed interactive Streamlit dashboard for live
transaction analysis
Handled class imbalance (0.17% fraud) using SMOTE
oversampling technique

## 👨‍💻 Author
**Dhairya Jani** — [GitHub](https://github.com/DhairyaJani2014)