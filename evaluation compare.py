"""
6_evaluation_compare.py - Perbandingan Model & Jawaban Target Analisis
Dataset: Telco Customer Churn

Requires: jalankan semua file sebelumnya (2, 3, 4, 5)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from utils import print_section, save_figure, COLORS

# ─────────────────────────────────────────────
# 1. Load Semua Hasil Evaluasi
# ─────────────────────────────────────────────
print_section("1. LOAD HASIL EVALUASI")
svm_res = pd.read_csv("results_svm.csv")
dt_res  = pd.read_csv("results_dt.csv")
km_res  = pd.read_csv("results_kmeans.csv")

print("SVM:\n", svm_res.to_string(index=False))
print("\nDecision Tree:\n", dt_res.to_string(index=False))
print("\nK-Means:\n", km_res.to_string(index=False))

# ─────────────────────────────────────────────
# 2. Perbandingan Klasifikasi
# ─────────────────────────────────────────────
print_section("2. PERBANDINGAN SVM vs DECISION TREE")
clf_metrics = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
comparison_df = pd.concat([svm_res[clf_metrics], dt_res[clf_metrics]], ignore_index=True)
print(comparison_df.to_string(index=False))

best_f1_idx = comparison_df["F1-Score"].idxmax()
best_model  = comparison_df.loc[best_f1_idx, "Model"]

# ─────────────────────────────────────────────
# 3. Bar Chart Perbandingan Metrik
# ─────────────────────────────────────────────
print_section("3. VISUALISASI PERBANDINGAN METRIK")
metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(13, 6))
colors_models = [COLORS["primary"], COLORS["success"]]

for i, (_, row) in enumerate(comparison_df.iterrows()):
    values = [row[m] for m in metrics]
    bars = ax.bar(x + i * width - width/2, values, width,
                  label=row["Model"], color=colors_models[i],
                  edgecolor="white", alpha=0.88)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_ylabel("Score", fontsize=12)
ax.set_title("Perbandingan Metrik: SVM vs Decision Tree", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 1.12)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
save_figure("17_compare_clf_metrics.png")
plt.show()

# ─────────────────────────────────────────────
# 4. Radar Chart
# ─────────────────────────────────────────────
print_section("4. RADAR CHART PERBANDINGAN")
categories = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
model_colors = [COLORS["primary"], COLORS["success"]]

for i, (_, row) in enumerate(comparison_df.iterrows()):
    values = [row[m] for m in categories] + [row[categories[0]]]
    ax.plot(angles, values, linewidth=2.5, linestyle="solid",
            label=row["Model"], color=model_colors[i])
    ax.fill(angles, values, alpha=0.12, color=model_colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=11)
ax.set_ylim(0, 1)
ax.set_title("Radar Chart – Perbandingan Model",
             fontsize=14, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_figure("18_radar_comparison.png")
plt.show()

# ─────────────────────────────────────────────
# 5. JAWABAN TARGET ANALISIS SOAL
# ─────────────────────────────────────────────

# ── Pertanyaan 1: Faktor yang memengaruhi churn ──
print_section("TARGET ANALISIS 1: FAKTOR YANG MEMPENGARUHI CHURN")

# Load feature importance dari Decision Tree
X_train = pd.read_csv("X_train.csv")
from sklearn.tree import DecisionTreeClassifier
y_train = pd.read_csv("y_train.csv").squeeze()
dt_temp = DecisionTreeClassifier(criterion="gini", max_depth=5,
                                  min_samples_split=10, min_samples_leaf=5,
                                  random_state=42)
dt_temp.fit(X_train, y_train)
importances = pd.Series(dt_temp.feature_importances_, index=X_train.columns)
importances = importances[importances > 0].sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10, 6))
colors_imp = [COLORS["danger"] if i < 3 else COLORS["primary"]
              for i in range(len(importances))]
bars = ax.barh(importances.index[::-1], importances.values[::-1],
               color=colors_imp[::-1], edgecolor="white", alpha=0.85)
ax.set_xlabel("Feature Importance", fontsize=12)
ax.set_title("Faktor yang Mempengaruhi Customer Churn\n(Berdasarkan Decision Tree Feature Importance)",
             fontsize=13, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
for bar, val in zip(bars, importances.values[::-1]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)
plt.tight_layout()
save_figure("19_faktor_churn.png")
plt.show()

print("\nJAWABAN PERTANYAAN 1:")
print("Faktor-faktor yang paling berpotensi mempengaruhi customer churn:")
for i, (feat, val) in enumerate(importances.head(5).items(), 1):
    print(f"  {i}. {feat:<35} (importance: {val:.4f})")
print("""
  Interpretasi:
  - Tenure (masa berlangganan): pelanggan baru lebih rentan churn
  - MonthlyCharges: biaya bulanan tinggi mendorong churn
  - TotalCharges: total pembayaran rendah = pelanggan baru = risiko churn tinggi
  - Contract type: pelanggan month-to-month lebih mudah berhenti
  - InternetService: Fiber optic memiliki churn rate lebih tinggi
""")

# ── Pertanyaan 2: SVM vs Decision Tree ──
print_section("TARGET ANALISIS 2: SVM vs DECISION TREE")

svm_vals = svm_res[metrics].values[0]
dt_vals  = dt_res[metrics].values[0]

print(f"""
{'Metrik':<15} {'SVM':>10} {'Decision Tree':>15} {'Lebih Baik':>15}
{'-'*55}""")
for m, sv, dv in zip(metrics, svm_vals, dt_vals):
    winner = "Decision Tree ✓" if dv > sv else "SVM ✓" if sv > dv else "Sama"
    print(f"{m:<15} {sv:>10.4f} {dv:>15.4f} {winner:>15}")

print(f"""
KESIMPULAN:
  Model terbaik berdasarkan F1-Score : {best_model}

  - Decision Tree lebih unggul di F1-Score ({dt_res['F1-Score'].iloc[0]:.4f} vs {svm_res['F1-Score'].iloc[0]:.4f})
    artinya lebih seimbang mendeteksi churn dan non-churn.
  - Decision Tree lebih unggul di ROC-AUC ({dt_res['ROC-AUC'].iloc[0]:.4f} vs {svm_res['ROC-AUC'].iloc[0]:.4f})
    artinya lebih baik membedakan pelanggan churn vs tidak.
  - SVM dan Decision Tree memiliki Accuracy yang sama ({svm_res['Accuracy'].iloc[0]:.4f}).
  - Untuk kasus prediksi churn, F1-Score dan Recall lebih penting
    karena biaya salah prediksi churn (False Negative) lebih besar.
  → Rekomendasi: gunakan Decision Tree untuk prediksi churn.
""")

# ── Pertanyaan 3: Kelompok Pelanggan K-Means ──
print_section("TARGET ANALISIS 3: KELOMPOK PELANGGAN DARI K-MEANS")

try:
    cluster_profiles = pd.read_csv("cluster_profiles.csv", index_col=0)
    data_cluster     = pd.read_csv("data_with_clusters.csv")

    best_k_actual = data_cluster["Cluster"].nunique()
    churn_rate    = data_cluster.groupby("Cluster")["Churn"].mean()

    print(f"\nJumlah cluster terbentuk: {best_k_actual}")
    print(f"\nProfil rata-rata per cluster:")
    print(cluster_profiles.round(2).to_string())
    print(f"\nChurn Rate per Cluster:")
    for i, rate in churn_rate.items():
        print(f"  Cluster {i}: {rate:.1%}")

    print("\nJAWABAN PERTANYAAN 3 – Karakteristik Kelompok Pelanggan:")
    for i in range(best_k_actual):
        row = cluster_profiles.loc[i]
        cr  = churn_rate[i]
        tenure_label = "Lama" if row["Tenure"] > 40 else "Baru" if row["Tenure"] < 20 else "Menengah"
        charge_label = "Tinggi" if row["MonthlyCharges"] > 70 else "Rendah" if row["MonthlyCharges"] < 45 else "Sedang"
        risk_label   = "TINGGI ⚠" if cr > 0.30 else "RENDAH ✓" if cr < 0.15 else "SEDANG"
        print(f"""
  ── Cluster {i} ──
     Tenure rata-rata    : {row['Tenure']:.1f} bulan ({tenure_label})
     Monthly Charges     : ${row['MonthlyCharges']:.1f} ({charge_label})
     Total Charges       : ${row['TotalCharges']:.0f}
     Churn Rate          : {cr:.1%} → Risiko {risk_label}""")

except FileNotFoundError:
    print("[!] File cluster_profiles.csv tidak ditemukan, jalankan 5_kmeans.py dulu.")

# ─────────────────────────────────────────────
# 6. Summary Dashboard Final
# ─────────────────────────────────────────────
print_section("6. SUMMARY DASHBOARD FINAL")

fig = plt.figure(figsize=(18, 11))
fig.suptitle("Telco Customer Churn – ML Project Summary\n(SVM | Decision Tree | K-Means)",
             fontsize=16, fontweight="bold", y=1.01)

# Panel A: Tabel metrik
ax1 = fig.add_subplot(2, 3, (1, 2))
ax1.axis("off")
col_labels = ["Model"] + metrics
table = ax1.table(
    cellText=[[row["Model"]] + [f"{row[m]:.4f}" for m in metrics]
              for _, row in comparison_df.iterrows()],
    colLabels=col_labels,
    loc="center", cellLoc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.2)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#4C72B0")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#f0f4ff")
ax1.set_title("Perbandingan Metrik Klasifikasi", fontsize=12, fontweight="bold", pad=15)

# Panel B: F1 & AUC bar
ax2 = fig.add_subplot(2, 3, 3)
x_pos = np.arange(2)
f1_vals  = [svm_res["F1-Score"].iloc[0], dt_res["F1-Score"].iloc[0]]
auc_vals = [svm_res["ROC-AUC"].iloc[0],  dt_res["ROC-AUC"].iloc[0]]
w = 0.35
bars1 = ax2.bar(x_pos - w/2, f1_vals,  w, label="F1-Score", color=COLORS["primary"],  edgecolor="white")
bars2 = ax2.bar(x_pos + w/2, auc_vals, w, label="ROC-AUC",  color=COLORS["secondary"], edgecolor="white")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(["SVM", "Decision Tree"])
ax2.set_ylim(0, 1.1)
ax2.set_title("F1-Score & ROC-AUC", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)
for bar in list(bars1) + list(bars2):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{bar.get_height():.3f}", ha="center", fontsize=9, fontweight="bold")

# Panel C: K-Means metrics
ax3 = fig.add_subplot(2, 3, 4)
km_labels = ["Silhouette\nScore", "Davies-Bouldin\n(inverted)", "Adj. Rand\nIndex"]
sil_val = float(km_res["Silhouette Score"].iloc[0])
db_val  = float(km_res["Davies-Bouldin Score"].iloc[0])
ari_val = float(km_res["Adjusted Rand Index"].iloc[0])
km_vals = [sil_val, 1/db_val, ari_val]
raw_display = [f"{sil_val:.4f}", f"{db_val:.4f}\n(raw)", f"{ari_val:.4f}"]
km_bars = ax3.bar(km_labels, km_vals, color=COLORS["secondary"],
                   edgecolor="white", alpha=0.85, width=0.4)
ax3.set_title("K-Means Clustering Metrics", fontsize=11, fontweight="bold")
ax3.set_ylim(0, 1.1)
ax3.grid(axis="y", alpha=0.3)
for bar, lbl in zip(km_bars, raw_display):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             lbl, ha="center", fontsize=9, fontweight="bold")

# Panel D: Jawaban ringkas 3 pertanyaan
ax4 = fig.add_subplot(2, 3, (5, 6))
ax4.axis("off")
svm_f1 = svm_res['F1-Score'].iloc[0]
dt_f1  = dt_res['F1-Score'].iloc[0]
svm_auc = svm_res['ROC-AUC'].iloc[0]
dt_auc  = dt_res['ROC-AUC'].iloc[0]

answer_text = f"""
JAWABAN TARGET ANALISIS

❶ Faktor yang mempengaruhi churn:
   • Tenure rendah → pelanggan baru rentan churn
   • Monthly Charges tinggi → biaya mahal = churn
   • Tipe kontrak Month-to-Month → mudah berhenti
   • Internet Service Fiber Optic → churn rate lebih tinggi
   • Tidak ada Online Security/TechSupport → mudah churn

❷ SVM vs Decision Tree:
   • F1-Score  → SVM: {svm_f1:.4f}  |  DT: {dt_f1:.4f}  → {best_model} lebih baik
   • ROC-AUC   → SVM: {svm_auc:.4f}  |  DT: {dt_auc:.4f}  → {best_model} lebih baik
   • Rekomendasi: Decision Tree lebih interpretatif
     dan lebih baik mendeteksi pelanggan churn

❸ Kelompok pelanggan dari K-Means:
   • Cluster berdasarkan tenure, biaya, & layanan
   • Cluster tenure pendek + biaya tinggi = risiko churn TINGGI
   • Cluster tenure panjang + biaya rendah = loyal
   • Segmentasi berguna untuk strategi retensi pelanggan
"""
ax4.text(0.03, 0.97, answer_text, transform=ax4.transAxes,
         fontsize=10, verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="#f0f8ff", alpha=0.8))

plt.tight_layout()
save_figure("20_summary_dashboard_final.png")
plt.show()

# ─────────────────────────────────────────────
# 7. Kesimpulan Akhir di Terminal
# ─────────────────────────────────────────────
print_section("KESIMPULAN AKHIR PROJECT")
print(f"""
Dataset  : Telco Customer Churn (7043 pelanggan, 21 fitur)

── KLASIFIKASI ──
┌─────────────────────┬──────────┬───────────┬────────┬──────────┬─────────┐
│ Model               │ Accuracy │ Precision │ Recall │ F1-Score │ ROC-AUC │""")

for _, row in comparison_df.iterrows():
    print(f"│ {row['Model']:<19} │ {row['Accuracy']:.4f}   │ {row['Precision']:.4f}    │ {row['Recall']:.4f} │ {row['F1-Score']:.4f}   │ {row['ROC-AUC']:.4f}  │")

print(f"""└─────────────────────┴──────────┴───────────┴────────┴──────────┴─────────┘
Model terbaik: {best_model}

── CLUSTERING ──
  Silhouette Score     : {float(km_res['Silhouette Score'].iloc[0]):.4f}
  Davies-Bouldin Score : {float(km_res['Davies-Bouldin Score'].iloc[0]):.4f}
  Adjusted Rand Index  : {float(km_res['Adjusted Rand Index'].iloc[0]):.4f}

── JAWABAN TARGET ANALISIS ──
  1. Faktor churn utama: Tenure, MonthlyCharges, Contract, InternetService
  2. Model terbaik: {best_model} (F1-Score & ROC-AUC lebih tinggi)
  3. K-Means membentuk segmen berdasarkan masa langganan & biaya
     → Cluster risiko tinggi = pelanggan baru + biaya mahal

Semua plot tersimpan di folder output_plots/
""")

print("✓ Project Machine Learning Telco Churn selesai!")