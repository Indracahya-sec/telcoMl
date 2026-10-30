"""
5_kmeans.py - Clustering dengan K-Means
Dataset: Telco Customer Churn

Segmentasi berdasarkan 5 fitur spesifik sesuai soal:
  - tenure, InternetService, PaymentMethod, MonthlyCharges, TotalCharges
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
from utils import print_section, save_figure, COLORS

# ─────────────────────────────────────────────
# 1. Load Data Asli
# ─────────────────────────────────────────────
print_section("1. LOAD DATA & PILIH 5 FITUR SEGMENTASI")
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Fix TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Encode InternetService
internet_map = {"No": 0, "DSL": 1, "Fiber optic": 2}
df["InternetService_enc"] = df["InternetService"].map(internet_map)

# Encode PaymentMethod
payment_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3
}
df["PaymentMethod_enc"] = df["PaymentMethod"].map(payment_map)

# Cek nilai unik yang tidak ter-mapping
print("Nilai unik InternetService:", df["InternetService"].unique())
print("Nilai unik PaymentMethod  :", df["PaymentMethod"].unique())

# Pilih 5 fitur
X = df[["tenure", "InternetService_enc", "PaymentMethod_enc",
        "MonthlyCharges", "TotalCharges"]].copy()
X.columns = ["Tenure", "InternetService", "PaymentMethod",
             "MonthlyCharges", "TotalCharges"]

y = df["Churn"].map({"Yes": 1, "No": 0})

# Imputasi NaN jika ada
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

print(f"\nShape: {X_imputed.shape}")
print(f"NaN setelah imputasi: {X_imputed.isnull().sum().sum()}")
print(X_imputed.describe().round(2))

# ─────────────────────────────────────────────
# 2. Scaling
# ─────────────────────────────────────────────
print_section("2. SCALING FITUR")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
print("StandardScaler diterapkan.")
print(f"NaN setelah scaling: {np.isnan(X_scaled).sum()}")

# ─────────────────────────────────────────────
# 3. Elbow Method
# ─────────────────────────────────────────────
print_section("3. ELBOW METHOD")
inertia_values = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia_values.append(km.inertia_)
    print(f"  K={k} | Inertia: {km.inertia_:.2f}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(k_range, inertia_values, marker="o", color=COLORS["primary"],
        linewidth=2.5, markersize=8, markerfacecolor="white", markeredgewidth=2)
ax.fill_between(k_range, inertia_values, alpha=0.08, color=COLORS["primary"])
ax.set_xlabel("Jumlah Cluster (K)", fontsize=12)
ax.set_ylabel("Inertia (WCSS)", fontsize=12)
ax.set_title("Elbow Method – Mencari K Optimal\n(Fitur: tenure, InternetService, PaymentMethod, MonthlyCharges, TotalCharges)",
             fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.set_xticks(list(k_range))
plt.tight_layout()
save_figure("12_kmeans_elbow.png")
plt.show()

# ─────────────────────────────────────────────
# 4. Silhouette Score
# ─────────────────────────────────────────────
print_section("4. SILHOUETTE SCORE")
silhouette_values = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_values.append(score)
    print(f"  K={k} | Silhouette Score: {score:.4f}")

best_k = list(k_range)[np.argmax(silhouette_values)]
print(f"\nK optimal: K={best_k}")

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(k_range, silhouette_values, color=COLORS["secondary"],
              edgecolor="white", alpha=0.85)
bars[np.argmax(silhouette_values)].set_color(COLORS["danger"])
ax.set_xlabel("Jumlah Cluster (K)", fontsize=12)
ax.set_ylabel("Silhouette Score", fontsize=12)
ax.set_title("Silhouette Score per K", fontsize=13, fontweight="bold")
ax.set_xticks(list(k_range))
ax.grid(axis="y", alpha=0.3)
ax.annotate(f"Best K={best_k}", xy=(best_k, max(silhouette_values)),
            xytext=(best_k + 0.5, max(silhouette_values) + 0.01),
            fontsize=11, color=COLORS["danger"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLORS["danger"]))
plt.tight_layout()
save_figure("13_kmeans_silhouette.png")
plt.show()

# ─────────────────────────────────────────────
# 5. Training K-Means
# ─────────────────────────────────────────────
print_section(f"5. TRAINING K-MEANS (K={best_k})")
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
cluster_labels = kmeans.fit_predict(X_scaled)

print("Jumlah data per cluster:")
unique, counts = np.unique(cluster_labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Cluster {u}: {c} data ({c/len(cluster_labels)*100:.1f}%)")

# ─────────────────────────────────────────────
# 6. Evaluasi
# ─────────────────────────────────────────────
print_section("6. EVALUASI CLUSTERING")
sil_score = silhouette_score(X_scaled, cluster_labels)
db_score  = davies_bouldin_score(X_scaled, cluster_labels)
ari_score = adjusted_rand_score(y, cluster_labels)

print(f"Silhouette Score    : {sil_score:.4f}")
print(f"Davies-Bouldin Score: {db_score:.4f}")
print(f"Adjusted Rand Index : {ari_score:.4f}")

# ─────────────────────────────────────────────
# 7. Visualisasi PCA 2D
# ─────────────────────────────────────────────
print_section("7. VISUALISASI CLUSTER (PCA 2D)")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_
print(f"Variance explained: PC1={explained_var[0]:.2%}, PC2={explained_var[1]:.2%}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("K-Means Clustering – Segmentasi Pelanggan (PCA 2D)",
             fontsize=14, fontweight="bold")

palette = sns.color_palette("Set2", best_k)
for i in range(best_k):
    mask = cluster_labels == i
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=[palette[i]], s=10, alpha=0.5, label=f"Cluster {i}")

centroids_pca = pca.transform(kmeans.cluster_centers_)
axes[0].scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                c="black", s=200, marker="X", zorder=10, label="Centroid")
axes[0].set_xlabel(f"PC1 ({explained_var[0]:.1%})")
axes[0].set_ylabel(f"PC2 ({explained_var[1]:.1%})")
axes[0].set_title("Hasil Clustering K-Means")
axes[0].legend(markerscale=2, fontsize=10)
axes[0].grid(True, alpha=0.3)

colors_true = [COLORS["success"] if c == 0 else COLORS["danger"] for c in y]
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=colors_true, s=10, alpha=0.4)
axes[1].set_xlabel(f"PC1 ({explained_var[0]:.1%})")
axes[1].set_ylabel(f"PC2 ({explained_var[1]:.1%})")
axes[1].set_title("Label Asli (No Churn / Churn)")
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=COLORS["success"], label="No Churn"),
                   Patch(facecolor=COLORS["danger"],  label="Churn")]
axes[1].legend(handles=legend_elements, fontsize=10)
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
save_figure("14_kmeans_pca_cluster.png")
plt.show()

# ─────────────────────────────────────────────
# 8. Profil Cluster
# ─────────────────────────────────────────────
print_section("8. PROFIL CLUSTER")
df_cluster = X_imputed.copy()
df_cluster["Cluster"] = cluster_labels
df_cluster["Churn"]   = y.values

profile   = df_cluster.groupby("Cluster")[["Tenure","MonthlyCharges","TotalCharges"]].mean().round(2)
churn_rate = df_cluster.groupby("Cluster")["Churn"].mean().round(3)

print("\nRata-rata fitur per cluster:")
print(profile)
print("\nChurn Rate per Cluster:")
for i, rate in churn_rate.items():
    print(f"  Cluster {i}: {rate:.1%}")

# ─────────────────────────────────────────────
# 9. Visualisasi Profil
# ─────────────────────────────────────────────
print_section("9. VISUALISASI PROFIL CLUSTER")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Profil Segmentasi Pelanggan per Cluster\n(K-Means – 5 Fitur Sesuai Soal)",
             fontsize=14, fontweight="bold")

palette_bar  = sns.color_palette("Set2", best_k)
cluster_names = [f"Cluster {i}" for i in range(best_k)]

# Tenure
axes[0,0].bar(cluster_names, profile["Tenure"], color=palette_bar, edgecolor="white")
axes[0,0].set_title("Rata-rata Tenure (Bulan)", fontweight="bold")
axes[0,0].set_ylabel("Bulan")
axes[0,0].grid(axis="y", alpha=0.3)
for i, v in enumerate(profile["Tenure"]):
    axes[0,0].text(i, v+0.5, f"{v:.1f}", ha="center", fontweight="bold")

# MonthlyCharges
axes[0,1].bar(cluster_names, profile["MonthlyCharges"], color=palette_bar, edgecolor="white")
axes[0,1].set_title("Rata-rata Monthly Charges ($)", fontweight="bold")
axes[0,1].set_ylabel("USD")
axes[0,1].grid(axis="y", alpha=0.3)
for i, v in enumerate(profile["MonthlyCharges"]):
    axes[0,1].text(i, v+0.5, f"${v:.1f}", ha="center", fontweight="bold")

# TotalCharges
axes[1,0].bar(cluster_names, profile["TotalCharges"], color=palette_bar, edgecolor="white")
axes[1,0].set_title("Rata-rata Total Charges ($)", fontweight="bold")
axes[1,0].set_ylabel("USD")
axes[1,0].grid(axis="y", alpha=0.3)
for i, v in enumerate(profile["TotalCharges"]):
    axes[1,0].text(i, v+10, f"${v:.0f}", ha="center", fontweight="bold")

# Churn Rate
churn_colors = [COLORS["danger"] if r == churn_rate.max()
                else COLORS["success"] if r == churn_rate.min()
                else COLORS["secondary"] for r in churn_rate.values]
axes[1,1].bar(cluster_names, churn_rate.values, color=churn_colors, edgecolor="white")
axes[1,1].set_title("Churn Rate per Cluster", fontweight="bold")
axes[1,1].set_ylabel("Churn Rate")
axes[1,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
axes[1,1].grid(axis="y", alpha=0.3)
for i, v in enumerate(churn_rate.values):
    axes[1,1].text(i, v+0.005, f"{v:.1%}", ha="center", fontweight="bold")

plt.tight_layout()
save_figure("15_kmeans_profil_cluster.png")
plt.show()

# ─────────────────────────────────────────────
# 10. Heatmap Profil
# ─────────────────────────────────────────────
print_section("10. HEATMAP PROFIL CLUSTER")
profile_full = profile.copy()
profile_full["ChurnRate"] = churn_rate
profile_norm = (profile_full - profile_full.min()) / (profile_full.max() - profile_full.min())

fig, ax = plt.subplots(figsize=(9, 5))
sns.heatmap(profile_norm.T, annot=profile_full.T.round(2),
            fmt=".2f", cmap="YlOrRd", linewidths=0.5, ax=ax)
ax.set_title("Heatmap Profil Cluster", fontsize=12, fontweight="bold")
ax.set_xlabel("Cluster")
plt.tight_layout()
save_figure("16_kmeans_heatmap_profil.png")
plt.show()

# ─────────────────────────────────────────────
# 11. Simpan Hasil
# ─────────────────────────────────────────────
print_section("11. SIMPAN HASIL")
pd.DataFrame([{
    "Model": f"K-Means (K={best_k})",
    "Silhouette Score": round(sil_score, 4),
    "Davies-Bouldin Score": round(db_score, 4),
    "Adjusted Rand Index": round(ari_score, 4)
}]).to_csv("results_kmeans.csv", index=False)

df_cluster.to_csv("data_with_clusters.csv", index=False)
profile_full.to_csv("cluster_profiles.csv")

print("results_kmeans.csv ✓")
print("data_with_clusters.csv ✓")
print("cluster_profiles.csv ✓")

print_section("K-MEANS SELESAI")