import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('data', exist_ok=True)
os.makedirs('plots', exist_ok=True)

df = pd.read_csv('data/loan_data.csv')

print("\n── STEP 1: First look ──")

# .shape gives you (rows, columns) as a tuple
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

print(df.head(5))

print(df.dtypes)

print("\n -- STEP 2: Missing values ──")

missing_count= df.isnull().sum()

missing_pct= (missing_count / len(df) *100).round(2)

missing_table = pd.DataFrame({'Missing Count': missing_count, 'Missing %': missing_pct})

missing_table = missing_table[missing_count > 0]

print(missing_table)

print("\n -- STEP 3: Statistical Summary --")

print(df.describe().round(2))

print("\n── STEP 4: Creating visualisation plots ──")

# Create a figure with 2 rows and 3 columns of charts
# figsize=(16, 10) sets the overall size in inches
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Loan Approval EDA', fontsize=16, fontweight='bold')

# ── Chart 1: How many approved vs rejected? ──
# value_counts() counts Y and N
status_counts = df['Loan_Status'].value_counts()
axes[0, 0].bar(
    ['Approved (Y)', 'Rejected (N)'],
    status_counts.values,
    color=['#4CAF50', '#F44336']
)
axes[0, 0].set_title('Loan Status Distribution')
axes[0, 0].set_ylabel('Count')

# ── Chart 2: Credit history vs approval ──
# groupby groups rows by Credit_History value
# value_counts(normalize=True) gives proportions not counts
# unstack() pivots from long to wide format
ch_approval = df.groupby('Credit_History')['Loan_Status']\
                .value_counts(normalize=True)\
                .unstack()
ch_approval.plot(
    kind='bar',
    ax=axes[0, 1],
    color=['#F44336', '#4CAF50'],
    rot=0
)
axes[0, 1].set_title('Approval Rate by Credit History')
axes[0, 1].set_xlabel('Credit History (0=Bad, 1=Good)')
axes[0, 1].set_ylabel('Proportion')

# ── Chart 3: Income distribution ──
# np.log1p = log(1 + x), handles zeros safely
# Without log: one bar at 3000-5000, tiny bars elsewhere
# With log: spread out, shows the real shape
axes[0, 2].hist(
    np.log1p(df['ApplicantIncome']),
    bins=40,
    color='#7B68EE'
)
axes[0, 2].set_title('Income Distribution (log scale)')
axes[0, 2].set_xlabel('Log(Income)')

# ── Chart 4: Loan amount by approval status ──
approved = df[df['Loan_Status'] == 'Y']['LoanAmount'].dropna()
rejected = df[df['Loan_Status'] == 'N']['LoanAmount'].dropna()
axes[1, 0].boxplot(
    [approved, rejected],
    labels=['Approved', 'Rejected'],
    patch_artist=True
)
axes[1, 0].set_title('Loan Amount: Approved vs Rejected')
axes[1, 0].set_ylabel('Loan Amount (₹ thousands)')

# ── Chart 5: Property area vs approval ──
area = df.groupby('Property_Area')['Loan_Status']\
         .value_counts()\
         .unstack()
area.plot(kind='bar', ax=axes[1, 1], rot=0,
          color=['#F44336', '#4CAF50'])
axes[1, 1].set_title('Approval by Property Area')

# ── Chart 6: Missing value map ──
# heatmap of True/False missing values
# Red = missing, white = present
sns.heatmap(
    df.isnull().astype(int),
    ax=axes[1, 2],
    cbar=False,
    yticklabels=False,
    cmap='Reds'
)
axes[1, 2].set_title('Missing Values (red = missing)')
axes[1, 2].tick_params(axis='x', rotation=45)

# Save the figure to disk
plt.tight_layout()
plt.savefig('plots/phase2_eda.png', dpi=120, bbox_inches='tight')
print("  Saved: plots/phase2_eda.png")
plt.close()


print("\n── STEP 5: Fixing missing values ──")

df_clean = df.copy()

text_columns =['Gender', 'Married', 'Dependents', 'Self_Employed']

for col in text_columns:
    most_common = df_clean[col].mode()[0]
    df_clean[col]= df_clean[col].fillna(most_common)
    print(f"{col}: filled {df[col].isnull().sum()} gaps with {most_common}")


number_columns = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']

for col in number_columns:
    median_value = df_clean[col].median()
    df_clean[col] = df_clean[col].fillna(median_value)
    print(f" {col} : filled {df[col].isnull().sum()} gaps with {median_value}")

    
df_clean['ApplicantIncome_log']   = np.log1p(df_clean['ApplicantIncome'])
df_clean['CoapplicantIncome_log'] = np.log1p(df_clean['CoapplicantIncome'])
df_clean['LoanAmount_log']        = np.log1p(df_clean['LoanAmount'])
df_clean.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount'], axis=1, inplace=True)

df_clean['Total_Income_log'] = np.log1p(np.expm1(df_clean['ApplicantIncome_log']) + np.expm1(df_clean['CoapplicantIncome_log']))

print("\n── STEP 7: Remove Loan_ID ──")

# Loan_ID is just a label like 'LP001002'
# It has ZERO predictive power
# Including it would add noise — model might 'learn'
# that LP001xxx loans approve more than LP002xxx (nonsense)
df_clean.drop('Loan_ID', axis=1, inplace=True)
print(f"  Removed Loan_ID. Final shape: {df_clean.shape}")

print("\n── STEP 8: Save clean data ──")

# Save to CSV so Phase 3 can load it
# index=False means don't save the row numbers as a column
df_clean.to_csv('data/loan_data_clean.csv', index=False)
print("  Saved: data/loan_data_clean.csv")

print("\nFinal columns in clean dataset:")
for col in df_clean.columns:
    print(f"  • {col}")

print("\n" + "=" * 60)
print("✓ PHASE 2 COMPLETE")
print("  Next: write phase 3 code")
print("=" * 60)






