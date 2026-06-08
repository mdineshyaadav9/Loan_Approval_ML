# ═══════════════════════════════════════════════════════════════
# PHASE 5 — Boosting the Score (Optimization)
# ═══════════════════════════════════════════════════════════════
#
# WHAT THIS FILE DOES:
#   Handle class imbalance, tune hyperparameters, use cross-validation,
#   and produce a final optimised model that's production-ready.
#
# WHY THIS STEP EXISTS:
#   Phase 4 gave us a decent model. This step takes it from
#   "student project" to "job interview ready".
#   This is where most real-world ML time is spent.
#
# RUN: python phase5_boost_score.py
# ═══════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report,
                              roc_auc_score, precision_recall_curve,
                              average_precision_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
import joblib, json, os

os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("=" * 60)
print("PHASE 5 — BOOSTING THE SCORE")
print("=" * 60)

# Load data
X_train = pd.read_csv('data/X_train.csv')
X_test  = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()
y_test  = pd.read_csv('data/y_test.csv').squeeze()
feature_names = X_train.columns.tolist()


# ══════════════════════════════════════════════════════════════
# PROBLEM 1 — Class Imbalance
# ══════════════════════════════════════════════════════════════
# THE PROBLEM:
#   Our dataset has ~70% Approved, ~30% Rejected.
#   A lazy model can score 70% accuracy by ALWAYS saying "Approved".
#   It never learns to identify rejections — which is what banks need!
#
# THE FIX — SMOTE (Synthetic Minority Oversampling TEchnique):
#   Instead of duplicating minority (Rejected) rows, SMOTE creates
#   NEW synthetic examples by interpolating between existing ones.
#   Result: balanced 50/50 dataset → model learns both classes equally.
#
# WHY NOT JUST DUPLICATE ROWS:
#   Duplicating causes overfitting on specific examples.
#   SMOTE creates diverse new samples → better generalisation.

print("\n── PROBLEM 1: Class Imbalance ──")
print(f"  Before SMOTE: Approved={y_train.sum()}, Rejected={(y_train==0).sum()}")

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"  After SMOTE:  Approved={y_train_bal.sum()}, Rejected={(y_train_bal==0).sum()}")
print("  ✓ Classes balanced — model will now learn rejections properly")


# ══════════════════════════════════════════════════════════════
# PROBLEM 2 — Single Train/Test Split is Unreliable
# ══════════════════════════════════════════════════════════════
# THE PROBLEM:
#   One 80/20 split might accidentally put "easy" examples in test.
#   Your model looks great — but it's just lucky.
#
# THE FIX — Cross-Validation (CV):
#   Split data into 5 "folds". Train on 4 folds, test on 1.
#   Repeat 5 times, each time using a different fold as test.
#   Final score = AVERAGE of 5 test scores.
#   Much more reliable than a single split.
#
#   Fold 1: [TEST][TRAIN][TRAIN][TRAIN][TRAIN]
#   Fold 2: [TRAIN][TEST][TRAIN][TRAIN][TRAIN]
#   ...

print("\n── PROBLEM 2: Cross-Validation (5-fold) ──")
rf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_base, X_train_bal, y_train_bal,
                             cv=cv, scoring='roc_auc')

print(f"  CV scores per fold: {[f'{s:.3f}' for s in cv_scores]}")
print(f"  Mean AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print("  Low std = model is consistent, not just lucky on one split")


# ══════════════════════════════════════════════════════════════
# PROBLEM 3 — Default Hyperparameters Aren't Optimal
# ══════════════════════════════════════════════════════════════
# WHAT ARE HYPERPARAMETERS:
#   Settings you choose BEFORE training. The model can't learn them.
#   Examples: how many trees? how deep? minimum samples per leaf?
#
# THE FIX — Grid Search:
#   Try every combination of settings, pick the best one.
#   GridSearchCV automates this with cross-validation.
#
#   n_estimators: more trees = more stable (but slower)
#   max_depth:    deeper = more complex patterns (but may overfit)
#   min_samples_split: higher = simpler rules (less overfitting)

print("\n── PROBLEM 3: Hyperparameter Tuning (Grid Search) ──")
param_grid = {
    'n_estimators':     [100, 200],
    'max_depth':        [10, 20, None],
    'min_samples_split':[2, 5, 10],
    'min_samples_leaf': [1, 2]
}

print("  Running Grid Search (this may take 1-2 minutes)...")
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train_bal, y_train_bal)

print(f"  Best parameters: {grid_search.best_params_}")
print(f"  Best CV AUC:     {grid_search.best_score_:.3f}")


# ══════════════════════════════════════════════════════════════
# FINAL MODEL — Optimised Random Forest
# ══════════════════════════════════════════════════════════════
best_rf = grid_search.best_estimator_
y_pred_final = best_rf.predict(X_test)
y_prob_final = best_rf.predict_proba(X_test)[:, 1]

acc_final = accuracy_score(y_test, y_pred_final)
auc_final = roc_auc_score(y_test, y_prob_final)

print(f"\n── Final Optimised Model Performance ──")
print(f"  Accuracy: {acc_final*100:.1f}%")
print(f"  ROC-AUC:  {auc_final:.3f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred_final, target_names=['Rejected','Approved']))


# ══════════════════════════════════════════════════════════════
# PROBLEM 4 — Default threshold (0.5) isn't always best
# ══════════════════════════════════════════════════════════════
# THE MODEL outputs a probability, e.g., 0.73 for approval.
# Default: if prob > 0.5 → Approve.
#
# But banks might prefer: if prob > 0.6 → Approve (more conservative)
# Or fintech might want: if prob > 0.4 → Approve (more inclusive)
#
# Precision = of those we approved, how many actually would repay?
# Recall    = of those who WOULD repay, how many did we approve?
# F1        = harmonic mean of precision and recall

print("\n── PROBLEM 4: Finding Optimal Decision Threshold ──")
precision, recall, thresholds = precision_recall_curve(y_test, y_prob_final)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_thresh_idx = f1_scores.argmax()
best_threshold = thresholds[best_thresh_idx] if best_thresh_idx < len(thresholds) else 0.5
print(f"  Default threshold (0.5): F1 = {f1_scores[np.abs(thresholds - 0.5).argmin() if len(thresholds)>0 else 0]:.3f}")
print(f"  Optimal threshold ({best_threshold:.2f}): F1 = {f1_scores[best_thresh_idx]:.3f}")

# Apply optimal threshold
y_pred_optimal = (y_prob_final >= best_threshold).astype(int)
print(f"  Accuracy with optimal threshold: {accuracy_score(y_test, y_pred_optimal)*100:.1f}%")


# ══════════════════════════════════════════════════════════════
# VISUALISATION — Optimisation results
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Phase 5 — Model Optimisation', fontsize=14, fontweight='bold')

# Plot 1: CV scores
axes[0].bar(range(1, 6), cv_scores, color='#7B68EE', edgecolor='white')
axes[0].axhline(cv_scores.mean(), color='red', linestyle='--', label=f'Mean={cv_scores.mean():.3f}')
axes[0].set_title('5-Fold Cross-Validation AUC')
axes[0].set_xlabel('Fold')
axes[0].set_ylabel('AUC Score')
axes[0].legend()

# Plot 2: Confusion Matrix
cm = confusion_matrix(y_test, y_pred_final)
im = axes[1].imshow(cm, cmap='Blues')
axes[1].set_xticks([0,1]); axes[1].set_yticks([0,1])
axes[1].set_xticklabels(['Predicted Reject','Predicted Approve'])
axes[1].set_yticklabels(['Actual Reject','Actual Approve'])
axes[1].set_title('Confusion Matrix — Final Model')
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, str(cm[i,j]), ha='center', va='center',
                     fontsize=18, color='white' if cm[i,j] > cm.max()/2 else 'black')

# Plot 3: Precision-Recall curve
axes[2].plot(recall, precision, color='#FF7F50', linewidth=2)
axes[2].axvline(x=recall[best_thresh_idx], color='green', linestyle='--', alpha=0.7,
                label=f'Optimal threshold={best_threshold:.2f}')
axes[2].set_xlabel('Recall (coverage of actual approvals)')
axes[2].set_ylabel('Precision (accuracy of approvals given)')
axes[2].set_title('Precision-Recall Curve')
axes[2].legend()

plt.tight_layout()
plt.savefig('plots/phase5_optimization.png', dpi=120, bbox_inches='tight')
print("\n  Saved: plots/phase5_optimization.png")
plt.close()


# ══════════════════════════════════════════════════════════════
# SAVE FINAL OPTIMISED MODEL
# ══════════════════════════════════════════════════════════════
joblib.dump(best_rf, 'models/best_model.pkl')

model_config = {
    'best_threshold':   float(best_threshold),
    'accuracy':         float(acc_final),
    'roc_auc':          float(auc_final),
    'best_params':      grid_search.best_params_,
    'feature_names':    feature_names
}
with open('models/model_config.json', 'w') as f:
    json.dump(model_config, f, indent=2)

print(f"\n── Saved final model ──")
print(f"  models/best_model.pkl       ← used by frontend")
print(f"  models/model_config.json    ← threshold + metadata")

print("\n" + "=" * 60)
print(f"✓ PHASE 5 COMPLETE — Final model: {acc_final*100:.1f}% accuracy, AUC={auc_final:.3f}")
print("  Next: Run the frontend with:  streamlit run frontend/app.py")
print("=" * 60)
