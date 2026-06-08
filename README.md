# 🏦 LoanIQ — AI-Powered Loan Approval System
### End-to-End Machine Learning Project | Finance Sector

---

## 📁 Project Structure

```
loan_approval_ml/
│
├── data/
│   ├── loan_data.csv           ← Raw dataset (614 loan applications)
│   └── loan_data_clean.csv     ← Created after Phase 2
│
├── models/
│   ├── best_model.pkl          ← Final optimised Random Forest
│   ├── scaler.pkl              ← StandardScaler (reused by frontend)
│   ├── model_config.json       ← Accuracy, threshold, parameters
│   └── feature_names.json     ← Column order for predictions
│
├── plots/
│   ├── phase2_eda.png          ← EDA visualisations
│   ├── phase4_models.png       ← Model comparison charts
│   └── phase5_optimization.png ← Optimisation results
│
├── frontend/
│   └── app.py                  ← Streamlit web application
│
├── phase2_eda_cleaning.py      ← Data exploration & fixing
├── phase3_preprocessing.py     ← Encoding, scaling, splitting
├── phase4_train_models.py      ← Train 3 ML models
├── phase5_boost_score.py       ← SMOTE, GridSearch, tuning
├── run_all.py                  ← Run all phases at once
├── requirements.txt            ← All Python dependencies
└── README.md                   ← This file
```

---

## 🚀 Setup Instructions

### Step 1 — Create a virtual environment (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 2 — Install all dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run all ML phases
```bash
python run_all.py
```
This runs all 4 phases in sequence (~3-5 minutes).

### Step 4 — Launch the web app
```bash
streamlit run frontend/app.py
```
Opens at **http://localhost:8501** in your browser.

---

## 📚 Learning Path (Run phases individually to learn step by step)

| Phase | File | What you learn |
|-------|------|----------------|
| 2 | `phase2_eda_cleaning.py` | EDA, missing values, outliers |
| 3 | `phase3_preprocessing.py` | Encoding, scaling, train/test split |
| 4 | `phase4_train_models.py` | Logistic Regression, Decision Tree, Random Forest |
| 5 | `phase5_boost_score.py` | SMOTE, Cross-validation, Grid Search |
| 6 | `frontend/app.py` | Streamlit frontend, live predictions |

---

## 🎯 ML Concepts Covered

- **Supervised Learning** — learning from labelled historical data
- **Binary Classification** — predict one of two outcomes (Approve/Reject)
- **Feature Engineering** — creating new features (Total_Income)
- **Label & One-Hot Encoding** — converting text to numbers
- **StandardScaler** — normalising feature ranges
- **Train/Test Split** — preventing data leakage
- **Logistic Regression** — probability-based linear classifier
- **Decision Tree** — rule-based flowchart model
- **Random Forest** — ensemble of 100 decision trees
- **SMOTE** — synthetic minority oversampling for class imbalance
- **Cross-Validation** — robust performance estimation
- **GridSearchCV** — automated hyperparameter tuning
- **ROC-AUC** — model evaluation beyond accuracy
- **Precision/Recall/F1** — class-specific performance metrics
- **Decision Threshold** — tuning approval sensitivity

---

## 🏦 Real-World Context

This project mirrors how banks like HDFC, ICICI, and fintech companies
like CRED and Paytm use ML for credit decisions:

1. Applicant submits details online
2. ML model scores the application in milliseconds
3. Score compared against risk threshold
4. Decision communicated instantly with reasons

---

## 💼 For Job Interviews

**Key talking points:**
- "I used Random Forest over Logistic Regression because..."
- "The dataset had class imbalance (70/30) which I handled with SMOTE"
- "I used cross-validation instead of a single split to get reliable metrics"
- "Credit History was the dominant feature — confirmed by feature importance"
- "I optimised the decision threshold to maximise F1 score, not just accuracy"

---

*Built as a hands-on ML learning project — Finance Sector*
