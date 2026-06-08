# ═══════════════════════════════════════════════════════════════
# PHASE 2 — Exploratory Data Analysis (EDA) & Data Cleaning
# ═══════════════════════════════════════════════════════════════
#
# WHAT THIS FILE DOES:
#   Load the raw dataset → find problems → fix them → save clean data
#
# WHY THIS STEP EXISTS:
#   ML models are mathematical — they cannot handle missing values,
#   text like "Male/Female", or wildly uneven number ranges.
#   Garbage in = garbage out. Clean data = better predictions.
#
# RUN: python phase2_eda_cleaning.py
# ═══════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Setup ──────────────────────────────────────────────────────
os.makedirs('data', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("=" * 60)
print("PHASE 2 — EDA & DATA CLEANING")
print("=" * 60)


# ══════════════════════════════════════════════════════════════
# STEP 1 — Load & first look
# ══════════════════════════════════════════════════════════════
# WHY: Before fixing anything, understand what you have.
# df = DataFrame — pandas' table structure (like Excel in Python)

df = pd.read_csv('data/loan_data.csv')

print("\n── STEP 1: First look at the data ──")
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())

print("\nData types of each column:")
print(df.dtypes)

# ── What dtypes mean ──
# object  → text/category (Gender, Married, etc.)
# int64   → whole numbers (ApplicantIncome)
# float64 → decimal numbers (LoanAmount, Credit_History)


# ══════════════════════════════════════════════════════════════
# STEP 2 — Find missing values
# ══════════════════════════════════════════════════════════════
# WHY: ML models crash or give wrong results with NaN (missing) values.
# We need to know WHERE the gaps are before we can fill them.

print("\n── STEP 2: Missing values ──")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
missing_df = missing_df[missing_df['Missing Count'] > 0]
print(missing_df)

# WHY each column has missing data (real-world reason):
# Gender        → applicant skipped optional field
# Married       → data entry error in bank system
# Dependents    → not declared by applicant
# Self_Employed → assumed "No" if not filled, but not recorded
# LoanAmount    → pending verification
# Loan_Amount_Term → bank offered flexible term, not yet decided
# Credit_History → no credit history exists (new to credit)


# ══════════════════════════════════════════════════════════════
# STEP 3 — Understand distributions (statistics)
# ══════════════════════════════════════════════════════════════
# WHY: We need to know if data is skewed, has outliers, or is
# unevenly distributed — all of these affect ML model quality.

print("\n── STEP 3: Statistical summary (numerical columns) ──")
print(df.describe().round(2))

# KEY THINGS TO NOTICE:
# ApplicantIncome: mean=~5400, max=81000 → BIG outliers!
#   A few people earn 10x more than average — this skews the model
# LoanAmount: mean~128, max=700 → also has outliers
# CoapplicantIncome: mean~1600, but 25th percentile=0
#   Many people have no co-applicant (income=0)


# ══════════════════════════════════════════════════════════════
# STEP 4 — Visualise key patterns
# ══════════════════════════════════════════════════════════════
# WHY: Humans understand charts better than numbers.
# These plots reveal patterns that guide our ML strategy.

print("\n── STEP 4: Generating visualisation plots ──")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Loan Approval EDA — Key Patterns', fontsize=16, fontweight='bold')

# Plot 1: Target variable distribution
# WHY CHECK THIS: If 99% are "Approved", model learns to always say
# "Approved" and gets 99% accuracy — but is useless. That's class imbalance.
status_counts = df['Loan_Status'].value_counts()
axes[0,0].bar(['Approved (Y)', 'Rejected (N)'], status_counts.values,
              color=['#4CAF50', '#F44336'], edgecolor='white', linewidth=0.5)
axes[0,0].set_title('Loan Status Distribution')
axes[0,0].set_ylabel('Count')
for i, v in enumerate(status_counts.values):
    axes[0,0].text(i, v + 5, f'{v}\n({v/len(df)*100:.1f}%)', ha='center', fontsize=10)

# Plot 2: Credit History vs Approval
# WHY: Credit history is our most powerful feature.
# This plot will SHOW you exactly how dominant it is.
ch_approval = df.groupby('Credit_History')['Loan_Status'].value_counts(normalize=True).unstack()
ch_approval.plot(kind='bar', ax=axes[0,1], color=['#F44336','#4CAF50'],
                 rot=0, edgecolor='white')
axes[0,1].set_title('Approval Rate by Credit History')
axes[0,1].set_xlabel('Credit History (0=Bad, 1=Good)')
axes[0,1].set_ylabel('Proportion')
axes[0,1].legend(['Rejected', 'Approved'])

# Plot 3: Income distribution (log scale to handle outliers)
# WHY LOG SCALE: If we plot raw income, the outliers (₹80k earners)
# squish everyone else into a tiny bar on the left — log scale spreads it out.
axes[0,2].hist(np.log1p(df['ApplicantIncome']), bins=40, color='#7B68EE', edgecolor='white')
axes[0,2].set_title('Applicant Income Distribution (log scale)')
axes[0,2].set_xlabel('Log(Income)')
axes[0,2].set_ylabel('Count')

# Plot 4: Loan Amount by Approval Status
# WHY BOXPLOT: Shows median, spread, and outliers side by side.
# We expect rejected loans to have higher amounts relative to income.
df_clean_plot = df.dropna(subset=['LoanAmount'])
approved_amounts = df_clean_plot[df_clean_plot['Loan_Status']=='Y']['LoanAmount']
rejected_amounts = df_clean_plot[df_clean_plot['Loan_Status']=='N']['LoanAmount']
axes[1,0].boxplot([approved_amounts, rejected_amounts], labels=['Approved','Rejected'],
                  patch_artist=True,
                  boxprops=dict(facecolor='#E8F5E9'),
                  medianprops=dict(color='#2E7D32', linewidth=2))
axes[1,0].set_title('Loan Amount: Approved vs Rejected')
axes[1,0].set_ylabel('Loan Amount (₹ thousands)')

# Plot 5: Property Area vs Approval
# WHY: Location of property affects collateral value and risk profile
area_approval = df.groupby('Property_Area')['Loan_Status'].value_counts().unstack()
area_approval.plot(kind='bar', ax=axes[1,1], color=['#F44336','#4CAF50'],
                   rot=0, edgecolor='white')
axes[1,1].set_title('Approval Rate by Property Area')
axes[1,1].set_ylabel('Count')
axes[1,1].legend(['Rejected', 'Approved'])

# Plot 6: Missing value heatmap
# WHY: Visual way to see patterns in missing data.
# If missing values cluster together, it might mean something systematic.
missing_matrix = df.isnull().astype(int)
sns.heatmap(missing_matrix, ax=axes[1,2], cbar=False,
            yticklabels=False, cmap='Reds')
axes[1,2].set_title('Missing Values Map (red = missing)')
axes[1,2].set_xticklabels(axes[1,2].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('plots/phase2_eda.png', dpi=120, bbox_inches='tight')
print("  Saved: plots/phase2_eda.png")
plt.close()


# ══════════════════════════════════════════════════════════════
# STEP 5 — Fix missing values (imputation)
# ══════════════════════════════════════════════════════════════
# WHY IMPUTATION INSTEAD OF DELETING ROWS:
#   If we delete every row with a missing value, we lose 100+ rows.
#   That's 16% of our data — bad for training. Better to fill smartly.
#
# STRATEGY:
#   Categorical columns (text) → fill with MODE (most common value)
#   Numerical columns (numbers) → fill with MEDIAN (not mean — median
#   is resistant to outliers; mean gets pulled by extreme values)

print("\n── STEP 5: Fixing missing values ──")

df_clean = df.copy()  # Always work on a copy — protect original data

# Categorical: fill with mode
# MODE = most frequent value in the column
cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed']
for col in cat_cols:
    mode_val = df_clean[col].mode()[0]
    df_clean[col].fillna(mode_val, inplace=True)
    print(f"  {col}: filled {df[col].isnull().sum()} missing with mode='{mode_val}'")

# Numerical: fill with median
# MEDIAN = middle value. e.g., [3, 5, 200] → median=5, mean=69
# Mean would be distorted by the 200. Median stays robust.
num_cols = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
for col in num_cols:
    median_val = df_clean[col].median()
    df_clean[col].fillna(median_val, inplace=True)
    print(f"  {col}: filled {df[col].isnull().sum()} missing with median={median_val}")

# Verify — no more missing values
remaining_missing = df_clean.isnull().sum().sum()
print(f"\n  Total missing values remaining: {remaining_missing} ✓")


# ══════════════════════════════════════════════════════════════
# STEP 6 — Remove Loan_ID (useless for ML)
# ══════════════════════════════════════════════════════════════
# WHY: Loan_ID is just a label. LP001002 vs LP001003 tells the model
# nothing about whether someone will repay. Including it would add
# noise and confuse the model.

df_clean.drop('Loan_ID', axis=1, inplace=True)
print(f"\n── STEP 6: Dropped Loan_ID. New shape: {df_clean.shape}")


# ══════════════════════════════════════════════════════════════
# STEP 7 — Handle outliers in income
# ══════════════════════════════════════════════════════════════
# WHY: Extreme values (e.g., someone earning ₹81,000/month) can pull
# the model toward unusual patterns. We cap outliers at the 99th
# percentile — a common real-world technique.
#
# METHOD: Log transformation
# Instead of using raw income (3000 vs 81000 = 27x difference),
# we take log (8.0 vs 11.3 = gentler gap). Model learns better.

print("\n── STEP 7: Handling income outliers with log transform ──")

df_clean['ApplicantIncome_log'] = np.log1p(df_clean['ApplicantIncome'])
df_clean['CoapplicantIncome_log'] = np.log1p(df_clean['CoapplicantIncome'])
df_clean['LoanAmount_log'] = np.log1p(df_clean['LoanAmount'])

# Drop original skewed columns (we'll use log versions)
df_clean.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount'], axis=1, inplace=True)
print("  Created log-transformed income and loan amount columns")


# ══════════════════════════════════════════════════════════════
# STEP 8 — Create a new feature: Total Income
# ══════════════════════════════════════════════════════════════
# WHY: Feature engineering — creating new columns from existing ones.
# The bank cares about TOTAL household income, not just applicant's.
# Total_Income = ApplicantIncome + CoapplicantIncome
# This single combined feature is often more predictive than both separate.

df_clean['Total_Income_log'] = np.log1p(
    np.expm1(df_clean['ApplicantIncome_log']) + np.expm1(df_clean['CoapplicantIncome_log'])
)
print("── STEP 8: Created Total_Income_log (combined household income)")


# ══════════════════════════════════════════════════════════════
# STEP 9 — Save cleaned data
# ══════════════════════════════════════════════════════════════

df_clean.to_csv('data/loan_data_clean.csv', index=False)
print(f"\n── STEP 9: Clean data saved → data/loan_data_clean.csv")
print(f"  Final shape: {df_clean.shape}")
print(f"\nFinal columns:")
for col in df_clean.columns:
    print(f"  • {col}")

print("\n" + "=" * 60)
print("✓ PHASE 2 COMPLETE — Data is clean and ready for ML")
print("  Next: Run phase3_preprocessing.py")
print("=" * 60)
