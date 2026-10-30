import subprocess
import sys

files = [
    "load explore .py",
    "preprocessing.py",
    "svm.py",
    "decision tree .py",
    "kmeans .py",
    "evaluation compare.py"
]

for f in files:
    print(f"\n{'='*50}\n>>> Running {f}\n{'='*50}")
    subprocess.run([sys.executable, f], check=True)
 
print("\n✓ Semua file selesai dijalankan!")