"""
4_decision_tree.py - Klasifikasi dengan Decision Tree
Dataset: Telco Customer Churn

Requires: jalankan 2_preprocessing.py terlebih dahulu
Input:  X_train.csv, X_test.csv, y_train.csv, y_test.csv  (TIDAK perlu scaled)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import cross_val_score
from utils import print_section, save_figure, COLORS

# ─────────────────────────────────────────────
# 1. Load Data (tidak perlu scaled untuk DT)
# ─────────────────────────────────────────────
print_section("1. LOAD DATA PREPROCESSED")
X_train = pd.read_csv("X_train.csv")
X_test  = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test  = pd.read_csv("y_test.csv").squeeze()

print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
print(f"Fitur: {X_train.columns.tolist()}")

# ─────────────────────────────────────────────
# 2. Bangun Model Decision Tree
# ─────────────────────────────────────────────
print_section("2. TRAINING DECISION TREE")
print("Criterion: Gini | Max Depth: 5 | Min Samples Split: 10")

dt_model = DecisionTreeClassifier(
    criterion="gini",       # Bisa juga "entropy"
    max_depth=5,            # Batasi kedalaman agar tidak overfit
    min_samples_split=10,   # Min sampel untuk split node
    min_samples_leaf=5,     # Min sampel di leaf
    random_state=42
)

dt_model.fit(X_train, y_train)
print(f"Model Decision Tree berhasil dilatih!")
print(f"Kedalaman pohon aktual: {dt_model.get_depth()}")
print(f"Jumlah leaves: {dt_model.get_n_leaves()}")

# ─────────────────────────────────────────────
# 3. Prediksi
# ─────────────────────────────────────────────
print_section("3. PREDIKSI")
y_pred       = dt_model.predict(X_test)
y_pred_proba = dt_model.predict_proba(X_test)[:, 1]

# ─────────────────────────────────────────────
# 4. Evaluasi Model
# ─────────────────────────────────────────────
print_section("4. EVALUASI MODEL DECISION TREE")
accuracy = accuracy_score(y_test, y_pred)
roc_auc  = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"ROC-AUC  : {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

# Cross Validation
print_section("4b. CROSS VALIDATION (5-Fold)")
cv_scores = cross_val_score(dt_model, X_train, y_train, cv=5, scoring="accuracy")
print(f"CV Scores  : {cv_scores.round(4)}")
print(f"CV Mean    : {cv_scores.mean():.4f}")
print(f"CV Std     : {cv_scores.std():.4f}")

# ─────────────────────────────────────────────
# 5. Confusion Matrix
# ─────────────────────────────────────────────
print_section("5. CONFUSION MATRIX")
cm = confusion_matrix(y_test, y_pred)
print(cm)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
            linewidths=1, linecolor="white",
            annot_kws={"size": 16, "weight": "bold"})
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title("Decision Tree - Confusion Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
save_figure("08_dt_confusion_matrix.png")
plt.show()

# ─────────────────────────────────────────────
# 6. ROC Curve
# ─────────────────────────────────────────────
print_section("6. ROC CURVE")
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color=COLORS["success"], lw=2.5,
        label=f"Decision Tree (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier")
ax.fill_between(fpr, tpr, alpha=0.08, color=COLORS["success"])
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("Decision Tree - ROC Curve", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_figure("09_dt_roc_curve.png")
plt.show()

# ─────────────────────────────────────────────
# 7. Visualisasi Pohon Keputusan
# ─────────────────────────────────────────────
print_section("7. VISUALISASI DECISION TREE")
fig, ax = plt.subplots(figsize=(22, 10))
plot_tree(
    dt_model,
    feature_names=X_train.columns.tolist(),
    class_names=["No Churn", "Churn"],
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax,
    impurity=True,
    proportion=False
)
ax.set_title("Decision Tree - Visualisasi Pohon Keputusan (Max Depth=5)",
             fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
save_figure("10_dt_tree_visualization.png")
plt.show()

# ─────────────────────────────────────────────
# 8. Feature Importance
# ─────────────────────────────────────────────
print_section("8. FEATURE IMPORTANCE")
importances = pd.Series(dt_model.feature_importances_, index=X_train.columns)
importances = importances.sort_values(ascending=False)
top_n = 15

print(f"Top {top_n} fitur paling penting:")
print(importances.head(top_n).to_string())

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(importances.head(top_n).index[::-1],
               importances.head(top_n).values[::-1],
               color=COLORS["success"], edgecolor="white", alpha=0.85)
ax.set_xlabel("Feature Importance (Gini)", fontsize=12)
ax.set_title(f"Decision Tree - Top {top_n} Feature Importance",
             fontsize=13, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
for bar, val in zip(bars, importances.head(top_n).values[::-1]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)
plt.tight_layout()
save_figure("11_dt_feature_importance.png")
plt.show()

# ─────────────────────────────────────────────
# 9. Simpan Hasil
# ─────────────────────────────────────────────
print_section("9. SIMPAN HASIL EVALUASI")
results = {
    "Model": "Decision Tree (Gini)",
    "Accuracy": round(accuracy, 4),
    "Precision": round(precision_score(y_test, y_pred), 4),
    "Recall": round(recall_score(y_test, y_pred), 4),
    "F1-Score": round(f1_score(y_test, y_pred), 4),
    "ROC-AUC": round(roc_auc, 4)
}

results_df = pd.DataFrame([results])
results_df.to_csv("results_dt.csv", index=False)
print("Hasil disimpan ke results_dt.csv")
print(results_df.to_string(index=False))

print_section("DECISION TREE SELESAI")
print("Jalankan 5_kmeans.py untuk clustering.")