# ═══════════════════════════════════════════════════════════════
# PHASE 4 — Train ML Models & Understand Why
# ═══════════════════════════════════════════════════════════════
#
# WHAT THIS FILE DOES:
#   Train 3 ML models, compare them, understand what each one
#   does internally, save the best model for the frontend.
#
# THE 3 MODELS (in order of complexity):
#   1. Logistic Regression — simple, fast, explainable
#   2. Decision Tree       — visual, rule-based, interpretable
#   3. Random Forest       — powerful ensemble of 100 trees
#
# RUN: python phase4_train_models.py
# ═══════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, roc_curve)
import joblib
import os

os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("=" * 60)
print("PHASE 4 — TRAINING ML MODELS")
print("=" * 60)

# ── Load preprocessed data ────────────────────────────────────
X_train = pd.read_csv('data/X_train.csv')
X_test  = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()
y_test  = pd.read_csv('data/y_test.csv').squeeze()

print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)} samples")
feature_names = X_train.columns.tolist()


# ══════════════════════════════════════════════════════════════
# MODEL 1 — Logistic Regression
# ══════════════════════════════════════════════════════════════
# HOW IT WORKS:
#   Despite the name "regression", this is a CLASSIFICATION model.
#   It calculates a probability between 0 and 1.
#   If P(approved) > 0.5 → Approve, else Reject.
#
# MATHEMATICALLY:
#   P(Y=1) = 1 / (1 + e^(-(w1*x1 + w2*x2 + ... + wn*xn)))
#   The model learns the weights (w) that best separate approvals from rejections.
#
# WHY START HERE:
#   ✓ Fast to train (seconds)
#   ✓ Very interpretable — weights show each feature's importance
#   ✓ Great baseline — if fancier models don't beat this, something's wrong
#   ✗ Assumes linear relationship between features and outcome

print("\n── MODEL 1: Logistic Regression ──")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]

acc_lr  = accuracy_score(y_test, y_pred_lr)
auc_lr  = roc_auc_score(y_test, y_prob_lr)
print(f"  Accuracy: {acc_lr*100:.1f}%")
print(f"  ROC-AUC:  {auc_lr:.3f}")

# Show what the model learned (feature weights)
print("\n  Feature weights (what the model learned):")
coefs = pd.Series(lr.coef_[0], index=feature_names).sort_values(key=abs, ascending=False)
for feat, coef in coefs.head(6).items():
    direction = "↑ Increases approval" if coef > 0 else "↓ Decreases approval"
    print(f"  {feat:<35} {coef:+.3f}  {direction}")


# ══════════════════════════════════════════════════════════════
# MODEL 2 — Decision Tree
# ══════════════════════════════════════════════════════════════
# HOW IT WORKS:
#   Like a flowchart of questions:
#   "Is Credit_History == 1?"
#     YES → "Is LoanAmount < 150?"
#              YES → APPROVE
#              NO  → "Is Income > 4000?" → ...
#     NO  → REJECT
#
#   The model learns which QUESTIONS to ask and in what ORDER
#   to best separate approvals from rejections.
#
# KEY TERM — Gini Impurity:
#   At each split, the tree asks: "Which question reduces confusion most?"
#   Gini = 0 means perfectly pure (all approved or all rejected)
#   Gini = 0.5 means maximum confusion (50/50 split)
#   The tree always picks the split that reduces Gini the most.
#
# WHY IT'S USEFUL HERE:
#   ✓ You can literally print the tree and read the rules
#   ✓ No scaling needed (not distance-based)
#   ✓ Handles non-linear relationships
#   ✗ Overfits easily — learns the training data TOO well

print("\n── MODEL 2: Decision Tree ──")
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
# max_depth=4 limits tree to 4 levels deep — prevents overfitting
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)[:, 1]

acc_dt = accuracy_score(y_test, y_pred_dt)
auc_dt = roc_auc_score(y_test, y_prob_dt)
print(f"  Accuracy: {acc_dt*100:.1f}%")
print(f"  ROC-AUC:  {auc_dt:.3f}")

# Print the actual decision rules
print("\n  The rules the tree learned (first 3 levels):")
tree_rules = export_text(dt, feature_names=feature_names, max_depth=3)
print("  " + tree_rules.replace("\n", "\n  ")[:800] + "...")


# ══════════════════════════════════════════════════════════════
# MODEL 3 — Random Forest
# ══════════════════════════════════════════════════════════════
# HOW IT WORKS:
#   Instead of ONE decision tree, it builds 100 trees.
#   Each tree is trained on a random SUBSET of data and features.
#   Final prediction = MAJORITY VOTE of all 100 trees.
#
# WHY THIS IS BETTER:
#   One tree might overfit by learning noise in data.
#   100 diverse trees vote → noise cancels out, signal remains.
#   This is called "ensemble learning" — wisdom of the crowd.
#
# THE KEY INSIGHT:
#   If tree 1 makes a mistake, trees 2-100 likely disagree.
#   When we take majority vote, the mistake gets outvoted.
#   Result: more stable, more accurate predictions.
#
# HYPERPARAMETERS (settings that control the model):
#   n_estimators=100  → how many trees to build
#   max_depth=None    → let trees grow fully (forest handles overfitting)
#   min_samples_split → how many samples needed to make a split

print("\n── MODEL 3: Random Forest ──")
rf = RandomForestClassifier(
    n_estimators=100,     # 100 trees
    max_depth=None,       # Trees grow fully
    min_samples_split=5,  # Need at least 5 samples to create a split
    random_state=42,
    n_jobs=-1             # Use all CPU cores
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

acc_rf = accuracy_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_prob_rf)
print(f"  Accuracy: {acc_rf*100:.1f}%")
print(f"  ROC-AUC:  {auc_rf:.3f}")


# ══════════════════════════════════════════════════════════════
# UNDERSTANDING THE METRICS
# ══════════════════════════════════════════════════════════════
# ACCURACY: Of all predictions, what % were correct?
#   Problem: If 70% are approved, always saying "Approve" = 70% accuracy
#   Accuracy alone is misleading for imbalanced data.
#
# CONFUSION MATRIX:
#   True Positive  (TP): Predicted Approve, Actually Approved → Correct ✓
#   True Negative  (TN): Predicted Reject, Actually Rejected  → Correct ✓
#   False Positive (FP): Predicted Approve, Actually Rejected → Bank loses money!
#   False Negative (FN): Predicted Reject, Actually Approved  → Lost customer!
#
# ROC-AUC (Area Under Curve):
#   Measures how well the model separates the two classes.
#   AUC = 1.0 → perfect model
#   AUC = 0.5 → random guessing
#   AUC > 0.80 → good model for finance

print("\n── Detailed report for best model (Random Forest) ──")
print(classification_report(y_test, y_pred_rf, target_names=['Rejected','Approved']))


# ══════════════════════════════════════════════════════════════
# VISUALISE — Model comparison + Feature importance
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Phase 4 — Model Training Results', fontsize=14, fontweight='bold')

# Plot 1: Accuracy comparison
models = ['Logistic\nRegression', 'Decision\nTree', 'Random\nForest']
accs   = [acc_lr, acc_dt, acc_rf]
aucs   = [auc_lr, auc_dt, auc_rf]
colors = ['#7B68EE', '#20B2AA', '#FF7F50']
bars = axes[0,0].bar(models, [a*100 for a in accs], color=colors, edgecolor='white', linewidth=0.5)
axes[0,0].set_title('Accuracy Comparison')
axes[0,0].set_ylabel('Accuracy (%)')
axes[0,0].set_ylim(0, 100)
for bar, acc in zip(bars, accs):
    axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{acc*100:.1f}%', ha='center', fontsize=11, fontweight='bold')

# Plot 2: ROC-AUC comparison
axes[0,1].bar(models, aucs, color=colors, edgecolor='white', linewidth=0.5)
axes[0,1].set_title('ROC-AUC Score (higher = better separator)')
axes[0,1].set_ylabel('AUC Score')
axes[0,1].set_ylim(0, 1.1)
axes[0,1].axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Good threshold (0.8)')
axes[0,1].legend()
for i, auc in enumerate(aucs):
    axes[0,1].text(i, auc + 0.01, f'{auc:.3f}', ha='center', fontsize=11, fontweight='bold')

# Plot 3: ROC Curves (all 3 models)
for prob, name, color in [(y_prob_lr,'Logistic Regression','#7B68EE'),
                           (y_prob_dt,'Decision Tree','#20B2AA'),
                           (y_prob_rf,'Random Forest','#FF7F50')]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc_val = roc_auc_score(y_test, prob)
    axes[1,0].plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC={auc_val:.3f})')
axes[1,0].plot([0,1],[0,1],'k--', alpha=0.3, label='Random guess (AUC=0.5)')
axes[1,0].set_xlabel('False Positive Rate (wrongly approved)')
axes[1,0].set_ylabel('True Positive Rate (correctly approved)')
axes[1,0].set_title('ROC Curves — How well each model separates classes')
axes[1,0].legend()

# Plot 4: Random Forest feature importance
# WHY FEATURE IMPORTANCE:
#   Random Forest tracks how much each feature reduced Gini impurity
#   across all 100 trees. Higher = more important for predictions.
feat_imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=True)
feat_imp.tail(10).plot(kind='barh', ax=axes[1,1], color='#FF7F50', edgecolor='white')
axes[1,1].set_title('Random Forest — Feature Importance')
axes[1,1].set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('plots/phase4_models.png', dpi=120, bbox_inches='tight')
print("\n  Saved: plots/phase4_models.png")
plt.close()


# ══════════════════════════════════════════════════════════════
# SAVE THE BEST MODEL
# ══════════════════════════════════════════════════════════════
# We save Random Forest — it has the best AUC and is robust.
# joblib saves Python objects to disk as .pkl (pickle) files.
# Later, the frontend loads this file and uses it for predictions.

best_model_name = 'Random Forest' if acc_rf >= max(acc_lr, acc_dt) else 'Logistic Regression'
joblib.dump(rf, 'models/random_forest.pkl')
joblib.dump(lr, 'models/logistic_regression.pkl')
joblib.dump(dt, 'models/decision_tree.pkl')

# Save feature names (needed to process frontend input in same order)
import json
with open('models/feature_names.json', 'w') as f:
    json.dump(feature_names, f)

print(f"\n── Saved all models to models/ directory ──")
print(f"  Best model: {best_model_name} (Accuracy: {max(acc_lr,acc_dt,acc_rf)*100:.1f}%)")

print("\n" + "=" * 60)
print("✓ PHASE 4 COMPLETE — Models trained and saved")
print("  Random Forest is our best model for the frontend")
print("  Next: Run phase5_boost_score.py")
print("=" * 60)
