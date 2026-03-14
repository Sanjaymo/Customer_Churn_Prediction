# 📡 Customer Churn Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-Classifier-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

> An interactive machine-learning web app to predict telecom customer churn using **XGBoost**, built with **Streamlit**. Features animated UI, real-time risk scoring, analytics dashboard, and downloadable prediction reports.

---

## 🚀 Features

- **XGBoost Classifier** trained on the Telco Customer Churn dataset
- Beautiful animated UI with light/dark mode adaptive colours
- 📊 **Analytics Dashboard** — churn distribution, feature importance, confusion matrix, probability histograms
- 🔮 **Real-time Prediction** — churn probability gauge, colour-coded risk result
- ⬇️ **Download Reports** — CSV and `.txt` prediction reports
- Sidebar customer profile form with dynamic encoding

---

## 🗂️ Project Structure

```
Costumers_Churn_Prediction/
│
├── streamlit_app.py                        # Main Streamlit application
├── Telco-Customer-Churn.csv      # Dataset (required)
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Sanjaymo/Costumers_Churn_Prediction.git
cd Costumers_Churn_Prediction
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the Dataset

Place the `Telco-Customer-Churn.csv` file in the **root directory** of the project.  
You can download it from [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

### 5. Run the App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 📦 Requirements

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.2.0
xgboost>=1.7.0
plotly>=5.14.0
```

Save the above as `requirements.txt` in the project root.

---

## 🧠 Model Details

| Parameter        | Value |
|-----------------|-------|
| Algorithm        | XGBClassifier |
| n_estimators     | 450 |
| learning_rate    | 0.01 |
| max_depth        | 3 |
| subsample        | 0.7 |
| colsample_bytree | 0.7 |
| gamma            | 0.3 |
| reg_lambda       | 2.0 |
| reg_alpha        | 0.5 |
| Test Split       | 20% |
| Stratified       | Yes |

---

## 📊 Dashboard Panels

| Panel | Description |
|-------|-------------|
| **Metric Cards** | Accuracy, Churn Rate, Precision, Recall at a glance |
| **Churn Distribution** | Donut chart showing churn vs retention split |
| **Probability Histogram** | Overlap of predicted probabilities per class |
| **Feature Importance** | Top-15 XGBoost feature importances (horizontal bar) |
| **Confusion Matrix** | Heatmap of true vs predicted labels |
| **Gauge Chart** | Real-time churn probability gauge per prediction |

---

## ⬇️ Download Options

After each prediction the app offers:
- **CSV Report** — input features + prediction + probability
- **Text Report** — full model metrics + prediction summary

---

## 📸 Screenshots

> *Add screenshots of your running app here.*

---

## 👤 Author

| | |
|---|---|
| **Name** | Sanjay Choudhari |
| **Phone** | +91 9963785768 |
| **Email** | [sanjaychoudhari288@gmail.com](mailto:sanjaychoudhari288@gmail.com) |
| **GitHub** | [github.com/Sanjaymo](https://github.com/Sanjaymo) |

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute.

---

<p align="center">Made with ❤️ by <strong>Sanjay Choudhari</strong></p>
