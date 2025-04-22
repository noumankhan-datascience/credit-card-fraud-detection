# Credit Card Transactions Fraud Detection 🔍💳

This project is a **machine learning pipeline** to detect fraudulent credit card transactions using the dataset from Kaggle. It uses preprocessing, feature engineering, and multiple classification algorithms to predict whether a transaction is fraudulent.

---

## 📊 Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- **Files Used:**
  - `fraudTrain.csv` – training dataset
  - `fraudTest.csv` – testing dataset
- **Size:** Over 1 million transaction records

---

## 📌 Features Used

| Feature        | Description                         |
|----------------|-------------------------------------|
| merchant       | Merchant name where the transaction occurred |
| category       | Transaction category (e.g., grocery, gas, etc.) |
| amt            | Amount of transaction               |
| city, state    | Location details of transaction     |
| trans_date_trans_time | Used to extract `hour`, `day`, and `month` |

---

## ⚙️ Technologies Used

- Python 🐍
- Scikit-learn 🔧
- Pandas, NumPy 🧮
- Streamlit (for Web UI) 🌐
- Git & GitHub 💻

---

## 🧠 Model Pipeline

- Categorical Encoding: `OneHotEncoder`
- Numerical Scaling: `StandardScaler`
- Model: `RandomForestClassifier`

You can also switch the model to:
- Logistic Regression
- Decision Tree
- etc.

---

## 📈 Evaluation Metrics

- Accuracy
- Precision, Recall
- F1-Score
- Confusion Matrix

---

## 🚀 Running the Streamlit App

```bash
streamlit run model.py
