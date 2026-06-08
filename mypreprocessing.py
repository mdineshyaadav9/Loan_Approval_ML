
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json

os.makedirs('models', exist_ok=True)

print("=" * 60)
print("PHASE 3 — PREPROCESSING")
print("=" * 60)

df=pd.read_csv('data/loan_data_clean.csv')

print(f" Loaded clean data: {df.shape}")

print(f" Columns available: ")
for col in df.columns:
    print(f" - {col}")


print("\n -- Step 1: Encoding text columns--")

