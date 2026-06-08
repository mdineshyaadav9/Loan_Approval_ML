# ═══════════════════════════════════════════════════════════════
# PHASE 3 — Data Preprocessing & Feature Engineering
# ═══════════════════════════════════════════════════════════════
#
# WHAT THIS FILE DOES:
#   Convert text categories → numbers, scale values, split into
#   train/test sets, prepare everything the ML model needs.
#
# WHY THIS STEP EXISTS:
#   ML models are math equations. Math can't compute with "Male"
#   or "Graduate" — only with numbers like 0 and 1.
#   This step is the translator between human data and ML math.
#
# RUN: python phase3_preprocessing.py
# ═══════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

os.makedirs('models', exist_ok=True)

print("=" * 60)
print("PHASE 3 — PREPROCESSING & FEATURE ENGINEERING")
print("=" * 60)

# ── Load clean data from Phase 2 ──────────────────────────────
df = pd.read_csv('data/loan_data_clean.csv')
print(f"\nLoaded clean data: {df.shape}")


# ══════════════════════════════════════════════════════════════
# STEP 1 — Encode categorical variables
# ══════════════════════════════════════════════════════════════
# WHY ENCODING:
#   "Male" vs "Female" → 1 vs 0
#   "Graduate" vs "Not Graduate" → 1 vs 0
#   This is called Label Encoding — converting labels to numbers.
#
# ANOTHER OPTION — One-Hot Encoding:
#   Property_Area has 3 values (Urban/Semiurban/Rural).
#   We can't use 0,1,2 because that implies Rural > Semiurban > Urban
#   which is false. Instead we create 3 binary columns:
#   Is_Urban? Is_Semiurban? Is_Rural?  → This is One-Hot Encoding.

print("\n── STEP 1: Encoding categorical variables ──")

df_encoded = df.copy()

# Binary columns (only 2 options) → Label Encoding
binary_cols = {
    'Gender':        {'Male': 1, 'Female': 0},
    'Married':       {'Yes': 1, 'No': 0},
    'Education':     {'Graduate': 1, 'Not Graduate': 0},
    'Self_Employed': {'Yes': 1, 'No': 0},
    'Loan_Status':   {'Y': 1, 'N': 0}   # Our target: 1=Approved, 0=Rejected
}

for col, mapping in binary_cols.items():
    df_encoded[col] = df_encoded[col].map(mapping)
    print(f"  {col}: {mapping}")

# Dependents: '0','1','2','3+' → 0,1,2,3
# WHY: 3+ is a string because of the '+'. We convert it to integer 3.
df_encoded['Dependents'] = df_encoded['Dependents'].replace('3+', 3).astype(int)
print(f"  Dependents: '3+' → 3 (integer)")

# Property_Area: 3 categories → One-Hot Encoding
# WHY ONE-HOT: No natural order between Urban/Semiurban/Rural.
# pd.get_dummies creates binary columns for each category.
# drop_first=True drops one column to avoid multicollinearity
# (if not Urban and not Rural → must be Semiurban — redundant)
df_encoded = pd.get_dummies(df_encoded, columns=['Property_Area'], drop_first=True)
print(f"  Property_Area → one-hot encoded (Property_Area_Semiurban, Property_Area_Urban)")

print(f"\n  Shape after encoding: {df_encoded.shape}")
print(f"  Columns: {list(df_encoded.columns)}")


# ══════════════════════════════════════════════════════════════
# STEP 2 — Separate features (X) and target (y)
# ══════════════════════════════════════════════════════════════
# WHY X and y:
#   X = inputs (everything we know about the applicant)
#   y = output (what we want to predict: 1=Approved, 0=Rejected)
#   The model learns: f(X) → y

print("\n── STEP 2: Separating features (X) and target (y) ──")

X = df_encoded.drop('Loan_Status', axis=1)
y = df_encoded['Loan_Status']

print(f"  X shape (features): {X.shape}")
print(f"  y shape (target):   {y.shape}")
print(f"  Class balance — Approved: {y.sum()} | Rejected: {(y==0).sum()}")


# ══════════════════════════════════════════════════════════════
# STEP 3 — Train/Test Split
# ══════════════════════════════════════════════════════════════
# WHY SPLIT THE DATA:
#   If you train AND test on the same data, the model just memorises
#   it — like studying with the answer key. It gives 99% accuracy
#   on training data but fails on real new applicants.
#
#   We split: 80% for training, 20% for testing (held-out data).
#   The model NEVER sees test data during training.
#
# stratify=y → ensures both splits have same % of Approved/Rejected.
# Without this, you might get all Rejected cases in test set by chance.

print("\n── STEP 3: Train/Test split (80/20) ──")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,       # 20% goes to test set
    random_state=42,      # Fixed seed = reproducible results
    stratify=y            # Keep class ratio same in both splits
)

print(f"  Training set:  {X_train.shape[0]} rows ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"  Test set:      {X_test.shape[0]} rows ({X_test.shape[0]/len(X)*100:.0f}%)")
print(f"  Train approved rate: {y_train.mean()*100:.1f}%")
print(f"  Test approved rate:  {y_test.mean()*100:.1f}%  ← should be ~same as train")


# ══════════════════════════════════════════════════════════════
# STEP 4 — Feature Scaling (Standardisation)
# ══════════════════════════════════════════════════════════════
# WHY SCALING:
#   ApplicantIncome ≈ 5,000–80,000
#   Credit_History  = 0 or 1
#
#   Without scaling, income completely dominates because its numbers
#   are thousands of times larger. The model ignores Credit_History
#   even though it's more predictive!
#
# StandardScaler: converts each feature to mean=0, std=1
# Formula: (value - mean) / std
# After scaling, all features are on the same playing field.
#
# IMPORTANT: Fit scaler ONLY on training data, then transform both.
# WHY: If we include test data in fitting, we "leak" test information
# into training — called data leakage, a common ML mistake.

print("\n── STEP 4: Feature scaling (StandardScaler) ──")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # Learn stats from train, then scale
X_test_scaled  = scaler.transform(X_test)         # Use SAME stats to scale test

print("  Applied StandardScaler (mean=0, std=1) to all features")
print(f"  Example — ApplicantIncome_log before: mean={X_train['ApplicantIncome_log'].mean():.2f}")

# Convert back to DataFrames (keeps column names)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=X_test.columns)


# ══════════════════════════════════════════════════════════════
# STEP 5 — Feature importance preview
# ══════════════════════════════════════════════════════════════
# WHY: Before training, we can get a sense of which features
# correlate most with loan approval. This guides our intuition.

print("\n── STEP 5: Correlation with Loan_Status ──")
corr = df_encoded.corr()['Loan_Status'].drop('Loan_Status').abs().sort_values(ascending=False)
print("  Top predictors (by correlation):")
for feat, val in corr.head(6).items():
    bar = '█' * int(val * 30)
    print(f"  {feat:<35} {bar} {val:.3f}")


# ══════════════════════════════════════════════════════════════
# STEP 6 — Save preprocessed data and scaler
# ══════════════════════════════════════════════════════════════
# WHY SAVE THE SCALER:
#   When the frontend receives a new application, it must scale
#   the input using the SAME parameters we learned here.
#   If we refit the scaler later, the numbers will be different
#   and the model's predictions will be wrong.

X_train_scaled.to_csv('data/X_train.csv', index=False)
X_test_scaled.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)
X_train.to_csv('data/X_train_raw.csv', index=False)  # Unscaled version for feature names

joblib.dump(scaler, 'models/scaler.pkl')
print(f"\n── STEP 6: Saved preprocessed data and scaler ──")
print("  data/X_train.csv, X_test.csv, y_train.csv, y_test.csv")
print("  models/scaler.pkl  ← reused by frontend for live predictions")

print("\n" + "=" * 60)
print("✓ PHASE 3 COMPLETE — Data is ML-ready")
print("  Next: Run phase4_train_models.py")
print("=" * 60)
