"""
2_preprocessing.py - Data Preprocessing & Feature Engineering
Dataset: Telco Customer Churn
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from utils import load_data, print_section

# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
print_section("1. LOAD DATA")
df = load_data()
print(f"Shape awal: {df.shape}")

# ─────────────────────────────────────────────
# 2. Drop Kolom Tidak Relevan
# ─────────────────────────────────────────────
print_section("2. DROP KOLOM TIDAK RELEVAN")
df.drop(columns=["customerID"], inplace=True)
print("Kolom 'customerID' dihapus.")

# ─────────────────────────────────────────────
# 3. Handle TotalCharges
# ─────────────────────────────────────────────
print_section("3. FIX KOLOM TotalCharges")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
n_missing = df["TotalCharges"].isnull().sum()
print(f"Missing setelah konversi: {n_missing} baris")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
print("Diisi dengan nilai median.")

# ─────────────────────────────────────────────
# 4. Encode Target (Churn)
# ─────────────────────────────────────────────
print_section("4. ENCODE TARGET")
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
print("Churn: Yes → 1, No → 0")
print(df["Churn"].value_counts())

# ─────────────────────────────────────────────
# 5. Encode Fitur Kategorik (Binary)
# ─────────────────────────────────────────────
print_section("5. ENCODE FITUR KATEGORIK")
binary_cols = [
    "gender", "Partner", "Dependents", "PhoneService",
    "PaperlessBilling", "MultipleLines", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies"
]

le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col].astype(str))
    print(f"  {col}: {df[col].unique()}")

# ─────────────────────────────────────────────
# 6. One-Hot Encoding
# ─────────────────────────────────────────────
print_section("6. ONE-HOT ENCODING")
ohe_cols = ["InternetService", "Contract", "PaymentMethod"]
df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)
print("Kolom setelah OHE:", df.columns.tolist())
print(f"Shape sekarang: {df.shape}")

# ─────────────────────────────────────────────
# 7. Pisah Fitur & Target
# ─────────────────────────────────────────────
print_section("7. SPLIT FITUR & TARGET")
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Pastikan semua kolom numerik
X = X.apply(pd.to_numeric, errors="coerce")

# Imputasi NaN dengan median
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
print(f"X shape: {X_imputed.shape}")
print(f"NaN setelah imputasi: {X_imputed.isnull().sum().sum()}")

# ─────────────────────────────────────────────
# 8. Train-Test Split
# ─────────────────────────────────────────────
print_section("8. TRAIN-TEST SPLIT (80:20)")
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train distribusi:\n{y_train.value_counts()}")
print(f"y_test distribusi:\n{y_test.value_counts()}")

# ─────────────────────────────────────────────
# 9. Scaling
# ─────────────────────────────────────────────
print_section("9. STANDARD SCALING")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df  = pd.DataFrame(X_test_scaled,  columns=X_test.columns)

# Final NaN check
print(f"NaN di X_train_scaled: {X_train_scaled_df.isnull().sum().sum()}")
print(f"NaN di X_test_scaled : {X_test_scaled_df.isnull().sum().sum()}")

# ─────────────────────────────────────────────
# 10. Simpan ke CSV
# ─────────────────────────────────────────────
print_section("10. SIMPAN DATA")

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

X_train_scaled_df.to_csv("X_train_scaled.csv", index=False)
X_test_scaled_df.to_csv("X_test_scaled.csv", index=False)

X_all_scaled = pd.DataFrame(scaler.transform(X_imputed), columns=X_imputed.columns)
X_all_scaled.to_csv("X_all_scaled.csv", index=False)
y.to_csv("y_all.csv", index=False)

print("Semua file berhasil disimpan!")
print_section("PREPROCESSING SELESAI")