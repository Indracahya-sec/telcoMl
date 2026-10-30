"""
utils.py - Fungsi-fungsi helper yang dipakai bersama antar file
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Path dataset
DATASET_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Warna konsisten untuk visualisasi
COLORS = {
    "primary": "#4C72B0",
    "secondary": "#DD8452",
    "success": "#55A868",
    "danger": "#C44E52",
    "palette": "Set2"
}

def load_data(path=DATASET_PATH):
    """Load dataset dari CSV"""
    df = pd.read_csv(path)
    print(f"[INFO] Dataset loaded: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df

def save_figure(filename, folder="output_plots"):
    """Simpan figure matplotlib ke folder"""
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"[INFO] Plot disimpan: {filepath}")

def print_section(title):
    """Cetak header section yang rapi"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)