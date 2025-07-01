# 💸 INR Price Prediction 📈

This project aims to build a predictive Machine Learning model to forecast fluctuations in INR (Indian Rupee) closing prices using historical currency exchange data.

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-ML%20Model-yellow?logo=scikit-learn"/>
  <img src="https://img.shields.io/badge/Status-Completed-success"/>
</div>

---

## 🚀 Project Objective

The goal is to build regression and classification models to:
- Predict the **next day's closing INR price** (Regression)
- Forecast **whether INR will rise or fall** (Classification)

---

## 🔍 Key Features

- 📊 **Linear Regression** for continuous INR closing price prediction  
- ✅ **Logistic Regression** for binary classification (up/down movement)  
- 🌲 **Random Forest Classifier** with hyperparameter tuning  
- ⚙️ **GridSearchCV** for model optimization  
- 🧹 Feature engineering and preprocessing pipeline  
- 📉 Evaluation using RMSE, Accuracy, Confusion Matrix

---

## 📁 Dataset

The dataset is synthetically generated for demonstration purposes and simulates historical INR/USD exchange rates over 500 days.

| Column Name | Description |
|-------------|-------------|
| `Open` | Opening INR value |
| `High` | Day's highest INR value |
| `Low` | Day's lowest INR value |
| `Close` | Day's closing INR value |
| `Next_Close` | Next day's closing INR value (Target for regression) |
| `Target` | 1 if `Next_Close > Close`, else 0 (Target for classification) |

---

## 🛠 Technologies Used

- **Python 3.11**
- **Pandas** – Data manipulation  
- **NumPy** – Numerical operations  
- **Scikit-learn** – ML models & tuning  
- **Matplotlib / Seaborn** (optional) – Visualization (not used in base script)

---

## 📈 Model Summary

| Model               | Task            | Metric        | Result         |
|--------------------|------------------|----------------|----------------|
| Linear Regression  | Regression       | RMSE           | ~1.44          |
| Logistic Regression| Classification   | Accuracy       | ~68%           |
| Random Forest       | Classification   | Accuracy       | ~70%           |
| Tuned RF (GridCV)   | Classification   | Accuracy       | ~71%           |

---

## 🧪 How to Run the Project

### 🔧 Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/INR-Price-Prediction.git
cd INR-Price-Prediction
