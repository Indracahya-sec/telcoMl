"""
1_load_explore.py - Load Dataset & Exploratory Data Analysis (EDA)
Dataset: Telco Customer Churn
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data, save_figure, print_section, COLORS

# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
print_section("1. LOAD DATASET")
df = load_data()

# ─────────────────────────────────────────────
# 2. Info Umum
# ─────────────────────────────────────────────
print_section("2. INFO UMUM")
print(df.info())
print("\nSampel data (5 baris pertama):")
print(df.head())

# ─────────────────────────────────────────────
# 3. Statistik Deskriptif
# ─────────────────────────────────────────────
print_section("3. STATISTIK DESKRIPTIF")
print(df.describe())

# ─────────────────────────────────────────────
# 4. Cek Missing Value
# ─────────────────────────────────────────────
print_section("4. MISSING VALUE")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Jumlah": missing, "Persentase (%)": missing_pct})
missing_df = missing_df[missing_df["Jumlah"] > 0]

if missing_df.empty:
    print("Tidak ada missing value secara eksplisit.")
    print("\n[!] Cek kolom TotalCharges (kemungkinan ada spasi kosong):")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    print(f"    Missing setelah konversi: {df['TotalCharges'].isnull().sum()}")
else:
    print(missing_df)

# ─────────────────────────────────────────────
# 5. Distribusi Target: Churn
# ─────────────────────────────────────────────
print_section("5. DISTRIBUSI TARGET (CHURN)")
churn_counts = df["Churn"].value_counts()
churn_pct = (churn_counts / len(df) * 100).round(2)
print(pd.DataFrame({"Jumlah": churn_counts, "Persentase (%)": churn_pct}))

# Plot pie + bar
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Distribusi Target: Churn", fontsize=15, fontweight="bold")

# Bar chart
axes[0].bar(churn_counts.index, churn_counts.values,
            color=[COLORS["success"], COLORS["danger"]], edgecolor="white", linewidth=1.5)
axes[0].set_title("Jumlah Customer")
axes[0].set_xlabel("Churn")
axes[0].set_ylabel("Jumlah")
for i, v in enumerate(churn_counts.values):
    axes[0].text(i, v + 30, str(v), ha="center", fontweight="bold")

# Pie chart
axes[1].pie(churn_counts.values, labels=churn_counts.index,
            autopct="%1.1f%%", colors=[COLORS["success"], COLORS["danger"]],
            startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("Proporsi Churn")

plt.tight_layout()
save_figure("01_distribusi_churn.png")
plt.show()

# ─────────────────────────────────────────────
# 6. Distribusi Fitur Numerik
# ─────────────────────────────────────────────
print_section("6. DISTRIBUSI FITUR NUMERIK")
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Distribusi Fitur Numerik", fontsize=14, fontweight="bold")

for i, col in enumerate(numeric_cols):
    axes[i].hist(df[col].dropna(), bins=30, color=COLORS["primary"], edgecolor="white", alpha=0.85)
    axes[i].set_title(col)
    axes[i].set_xlabel("Nilai")
    axes[i].set_ylabel("Frekuensi")

plt.tight_layout()
save_figure("02_distribusi_numerik.png")
plt.show()

# ─────────────────────────────────────────────
# 7. Korelasi Fitur Numerik
# ─────────────────────────────────────────────
print_section("7. HEATMAP KORELASI")
fig, ax = plt.subplots(figsize=(7, 5))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues",
            linewidths=0.5, ax=ax, square=True)
ax.set_title("Korelasi Fitur Numerik", fontsize=13, fontweight="bold")
plt.tight_layout()
save_figure("03_heatmap_korelasi.png")
plt.show()

# ─────────────────────────────────────────────
# 8. Boxplot Fitur Numerik vs Churn
# ─────────────────────────────────────────────
print_section("8. BOXPLOT FITUR NUMERIK vs CHURN")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Fitur Numerik vs Churn", fontsize=14, fontweight="bold")

for i, col in enumerate(numeric_cols):
    sns.boxplot(x="Churn", y=col, data=df,
                palette=[COLORS["success"], COLORS["danger"]], ax=axes[i])
    axes[i].set_title(col)

plt.tight_layout()
save_figure("04_boxplot_vs_churn.png")
plt.show()

# ─────────────────────────────────────────────
# 9. Fitur Kategorik vs Churn (beberapa kolom)
# ─────────────────────────────────────────────
print_section("9. FITUR KATEGORIK vs CHURN")
cat_cols = ["gender", "SeniorCitizen", "Partner", "Dependents",
            "PhoneService", "InternetService", "Contract", "PaymentMethod"]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Fitur Kategorik vs Churn", fontsize=15, fontweight="bold")
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    ct = pd.crosstab(df[col], df["Churn"], normalize="index") * 100
    ct.plot(kind="bar", ax=axes[i], color=[COLORS["success"], COLORS["danger"]],
            edgecolor="white", rot=30)
    axes[i].set_title(col)
    axes[i].set_ylabel("% Churn")
    axes[i].legend(["No", "Yes"], loc="upper right", fontsize=8)

plt.tight_layout()
save_figure("05_kategorik_vs_churn.png")
plt.show()

print_section("EDA SELESAI")
print("Semua plot disimpan di folder output_plots/")
print("Jalankan 2_preprocessing.py untuk tahap selanjutnya.")