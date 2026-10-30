"""
3_svm.py - Klasifikasi dengan Support Vector Machine (SVM)
Dataset: Telco Customer Churn

Requires: jalankan 2_preprocessing.py terlebih dahulu
Input:  X_train_scaled.csv, X_test_scaled.csv, y_train.csv, y_test.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, roc_curve
)
from utils import print_section, save_figure, COLORS

# ─────────────────────────────────────────────
# 1. Load Data (Hasil Preprocessing)
# ─────────────────────────────────────────────
print_section("1. LOAD DATA PREPROCESSED")
X_train = pd.read_csv("X_train_scaled.csv")
X_test  = pd.read_csv("X_test_scaled.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test  = pd.read_csv("y_test.csv").squeeze()

print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
print(f"y_train distribusi:\n{y_train.value_counts()}")

# ─────────────────────────────────────────────
# 2. Bangun Model SVM
# ─────────────────────────────────────────────
print_section("2. TRAINING SVM")
print("Kernel: RBF | C=1.0 | probability=True")

svm_model = SVC(
    kernel="rbf",       # Radial Basis Function kernel
    C=1.0,              # Parameter regularisasi
    gamma="scale",      # Otomatis berdasarkan fitur
    probability=True,   # Untuk ROC-AUC
    random_state=42
)

svm_model.fit(X_train, y_train)
print("Model SVM berhasil dilatih!")

# ─────────────────────────────────────────────
# 3. Prediksi
# ─────────────────────────────────────────────
print_section("3. PREDIKSI")
y_pred       = svm_model.predict(X_test)
y_pred_proba = svm_model.predict_proba(X_test)[:, 1]

# ─────────────────────────────────────────────
# 4. Evaluasi Model
# ─────────────────────────────────────────────
print_section("4. EVALUASI MODEL SVM")

accuracy = accuracy_score(y_test, y_pred)
roc_auc  = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"ROC-AUC  : {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

# ─────────────────────────────────────────────
# 5. Confusion Matrix
# ─────────────────────────────────────────────
print_section("5. CONFUSION MATRIX")
cm = confusion_matrix(y_test, y_pred)
print(cm)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
            linewidths=1, linecolor="white",
            annot_kws={"size": 16, "weight": "bold"})
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title("SVM - Confusion Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
save_figure("06_svm_confusion_matrix.png")
plt.show()

# ─────────────────────────────────────────────
# 6. ROC Curve
# ─────────────────────────────────────────────
print_section("6. ROC CURVE")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color=COLORS["primary"], lw=2.5,
        label=f"SVM (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier")
ax.fill_between(fpr, tpr, alpha=0.08, color=COLORS["primary"])
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("SVM - ROC Curve", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_figure("07_svm_roc_curve.png")
plt.show()

# ─────────────────────────────────────────────
# 7. Simpan Hasil Evaluasi
# ─────────────────────────────────────────────
print_section("7. SIMPAN HASIL EVALUASI")
from sklearn.metrics import precision_score, recall_score, f1_score

results = {
    "Model": "SVM (RBF)",
    "Accuracy": round(accuracy, 4),
    "Precision": round(precision_score(y_test, y_pred), 4),
    "Recall": round(recall_score(y_test, y_pred), 4),
    "F1-Score": round(f1_score(y_test, y_pred), 4),
    "ROC-AUC": round(roc_auc, 4)
}

results_df = pd.DataFrame([results])
results_df.to_csv("results_svm.csv", index=False)
print("Hasil disimpan ke results_svm.csv")
print(results_df.to_string(index=False))

print_section("SVM SELESAI")
print("Jalankan 4_decision_tree.py untuk model selanjutnya.")