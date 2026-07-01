# 💳 Credit Card Fraud Detection System

### End-to-end fraud detection using XGBoost Pipeline + Neural Networks (TensorFlow)

---

## 📌 Project Overview

A complete machine learning system for detecting fraudulent credit card
transactions — built on 284,807 real transactions with a severe class
imbalance (only 0.17% fraud). The project demonstrates a full ML workflow
from EDA through model comparison, covering every concept from the Stanford
ML Specialization (Advanced Learning Algorithms).

---

## 🛠️ Tech Stack

- **Python** — core language
- **Scikit-learn** — Pipeline, StandardScaler, StratifiedKFold, cross validation
- **XGBoost** — gradient boosted tree ensemble (primary model)
- **TensorFlow / Keras** — Neural Network (secondary model)
- **imbalanced-learn** — SMOTE for synthetic minority oversampling
- **Pandas / NumPy** — data wrangling and preprocessing
- **Matplotlib / Seaborn** — visualisation
- **Google Colab** — development environment (T4 GPU)

---

## 🔄 Pipeline Architecture

```
Raw Data (284,807 transactions)
        ↓
EDA — class imbalance, feature correlations, amount/time distributions
        ↓
Train / Validation / Test Split (Stratified — preserves fraud ratio)
        ↓
XGBoost Pipeline:
  StandardScaler → SMOTE → XGBClassifier (300 trees, depth=6)
        ↓
Neural Network (TensorFlow):
  Dense(256, ReLU) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(2, Softmax)
        ↓
Cross Validation (5-Fold StratifiedKFold)
        ↓
Bias/Variance Analysis (Learning Curves)
        ↓
Error Analysis (Confusion Matrix, ROC, Precision-Recall)
        ↓
Final Model Comparison Report
```

---

## 📊 Key Results

| Metric | XGBoost Pipeline | Neural Network |
|---|---|---|
| ROC AUC | 0.97+ | 0.96+ |
| Preprocessing | Automated Pipeline | Manual |
| Training Speed | Faster | Slower |
| Interpretability | Feature Importance | Limited |
| Fraud Recall | High | High |

---

## 🧠 ML Concepts Demonstrated

| Concept | Implementation |
|---|---|
| Neural Networks & Layers | TensorFlow Sequential — 256→128→64→2 |
| ReLU Activation | All hidden layers |
| Softmax Output | 2-neuron output — P(legit) + P(fraud) = 1 |
| Advanced Optimization | Adam optimizer with learning rate 0.001 |
| Multiclass Classification | Softmax output layer |
| Cross Validation | 5-Fold StratifiedKFold on training set |
| Bias / Variance Analysis | Learning curves — training vs validation |
| Error Analysis | Confusion matrix + Precision-Recall curve |
| Decision Tree → XGBoost | Gradient boosted ensemble (300 trees) |
| Scikit-learn Pipeline | StandardScaler + SMOTE + XGBClassifier |

---

## ⚖️ The Business Problem

Standard accuracy is misleading for fraud detection:

> A model that predicts **every transaction as legitimate**
> achieves **99.83% accuracy** but catches **zero fraud.**

This project addresses the real business objective:
- **Maximise fraud caught** (True Positive Rate / Recall)
- **Minimise false alarms** (False Positive Rate / Precision)
- The Precision-Recall curve visualises this tradeoff explicitly

---

## 🚀 How to Run

### Step 1 — Download the dataset
Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
→ Download `creditcard.csv`

### Step 2 — Open in Google Colab
Upload `creditcard.csv` to Colab files panel

### Step 3 — Install dependencies
```python
!pip install xgboost imbalanced-learn tensorflow -q
```

### Step 4 — Run cells in order
Run Cell 1 through Cell 13 sequentially

---

## 💡 Key Learnings

1. **Class imbalance is the central challenge** — SMOTE and
   `scale_pos_weight` both needed to handle 0.17% fraud rate
2. **Pipeline prevents data leakage** — scaler fitted only on
   training data, never on validation or test
3. **ROC AUC vs Accuracy** — AUC is the correct metric for
   imbalanced classification, not raw accuracy
4. **Precision-Recall tradeoff** — every threshold decision has
   a business cost (missed fraud vs false alarms)
5. **Early stopping** — prevented Neural Network overfitting
   without manual epoch tuning
6. **XGBoost Feature Importance** — V14, V17 emerged as the
   strongest fraud signals in the anonymised dataset

---

## 🔗 Related Projects

- [GAN vs VAE Generative Model Comparison](https://github.com/mohitramtekkar5-oss/GAN-vs-VAE-Comparison)
- [Walmart Data Chatbot](https://github.com/mohitramtekkar5-oss/Walmart-Data-Chatbot)
- [AI-Powered Data Insights](https://github.com/mohitramtekkar5-oss/AI-Powered-Data-Insights)
- [Walmart Sales Forecasting](https://github.com/mohitramtekkar5-oss/Walmart-Sales-Prediction-ML)

---

## 👤 Author

**Mohit Priya Ramtekkar**
Data Analyst | ML & Generative AI Enthusiast
[LinkedIn](https://linkedin.com/in/mohit-priya-ramtekkar) •
[GitHub](https://github.com/mohitramtekkar5-oss)

---
*Built as a hands-on implementation exercise applying concepts from
the Advanced Learning Algorithms course — Stanford University &
DeepLearning.AI on Coursera.*
