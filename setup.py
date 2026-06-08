import os
import subprocess
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import json

def setup():
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'models', 'best_model.pkl'
    )

    if os.path.exists(model_path):
        print("Model already exists — skipping")
        return

    print("Training model for cloud deployment...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'data'), exist_ok=True)

    # ── Load and clean data ──────────────────────────────────
    df = pd.read_csv(os.path.join(base_dir, 'data', 'loan_data.csv'))

    # Fix missing values
    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed']:
        df[col].fillna(df[col].mode()[0], inplace=True)
    for col in ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']:
        df[col].fillna(df[col].median(), inplace=True)

    # Log transform
    df['ApplicantIncome_log']   = np.log1p(df['ApplicantIncome'])
    df['CoapplicantIncome_log'] = np.log1p(df['CoapplicantIncome'])
    df['LoanAmount_log']        = np.log1p(df['LoanAmount'])
    df['Total_Income_log']      = np.log1p(df['ApplicantIncome'] + df['CoapplicantIncome'])

    # Drop originals
    df.drop(['Loan_ID','ApplicantIncome','CoapplicantIncome','LoanAmount'], axis=1, inplace=True)

    # ── Encode ───────────────────────────────────────────────
    mappings = {
        'Gender':        {'Male':1,'Female':0},
        'Married':       {'Yes':1,'No':0},
        'Education':     {'Graduate':1,'Not Graduate':0},
        'Self_Employed': {'Yes':1,'No':0},
        'Loan_Status':   {'Y':1,'N':0}
    }
    for col, mapping in mappings.items():
        df[col] = df[col].map(mapping)

    df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)
    df = pd.get_dummies(df, columns=['Property_Area'], drop_first=True)

    # ── Split ────────────────────────────────────────────────
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Scale ────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)

    # ── Train fast Random Forest (no GridSearch) ─────────────
    # Using best known parameters directly — skips GridSearch
    # which takes too long on free cloud servers
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # ── Evaluate ─────────────────────────────────────────────
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"Accuracy: {acc*100:.1f}%  AUC: {auc:.3f}")

    # ── Save everything ──────────────────────────────────────
    joblib.dump(model,  os.path.join(base_dir, 'models', 'best_model.pkl'))
    joblib.dump(scaler, os.path.join(base_dir, 'models', 'scaler.pkl'))

    feature_names = X_train.columns.tolist()
    with open(os.path.join(base_dir, 'models', 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f)

    config = {
        'best_threshold': 0.61,
        'accuracy': float(acc),
        'roc_auc':  float(auc),
        'feature_names': feature_names
    }
    with open(os.path.join(base_dir, 'models', 'model_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print("Model saved successfully!")

if __name__ == "__main__":
    setup()
