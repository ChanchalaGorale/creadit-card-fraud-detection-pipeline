# 💳 Credit Card Fraud Detection Pipeline

## A production-grade, modular Machine Learning pipeline to detect fraudulent credit card transactions.

## ⚙️ Features

- 📊 **Modular EDA** and preprocessing
- 🧼 Scalable and clean **data transformation pipelines**
- 📦 **Model training, evaluation, and serialization**
- 🔄 Real-time data **appending**
- 🧪 MLflow **experiment tracking**

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Create virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the Dataset

Download the credit card fraud detection dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it under:

```
data/raw/data.csv
```

---

## 📈 MLflow Tracking

Launch the MLflow UI locally:

```bash
mlflow ui
```

Access via: [http://localhost:5000](http://localhost:5000)

Tracks:

- Experiment name, run ID
- Parameters, metrics (accuracy, recall, etc.)
- Artifacts (model, plots)

---

## 🧪 Evaluation Metrics

- ✅ Accuracy
- 📉 Precision & Recall (emphasizing **Recall** for fraud detection)
- 🔁 F1-score
- 📊 AUC-ROC
