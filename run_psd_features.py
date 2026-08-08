"""
run_psd_features.py
--------------------
Local (VSCode) version of Complete_analysis_of_PSD_features.ipynb.

What changed vs. the notebook:
  - Removed the Google Colab drive-mount cell and the `!pip install`
    cells (install the packages once in a terminal instead, see below).
  - File paths now point at your local "data" folder instead of
    "/content/drive/MyDrive/...".
  - train_model_for_a_class() / train_model() are UNCHANGED — imported
    from common_train.py exactly as they were in the notebook.

One-time setup in a terminal:
    pip install xgboost imbalanced-learn scikit-learn numpy pandas

Run:
    python run_psd_features.py
"""

from pathlib import Path
from common_train import train_model, resolve_file

# ---------------------------------------------------------------------
# Point this at the "data" folder shown in your screenshot:
# This PC > Downloads > data dronerf > data
# (Path.home() resolves automatically to your Windows user folder,
#  so this does not need editing unless your folder is somewhere else.)
# ---------------------------------------------------------------------
DATA_DIR = Path.home() / "Downloads" / "data dronerf" / "data"


# ===================================================================
# Files that ARE currently in your data folder
# ===================================================================

print(">>> PSD (general)")
train_model(resolve_file(DATA_DIR, "psd_both_bands"), use_smote=True, average='weighted')

print(">>> PSD both bands, M=2048, L=1e5")
train_model(resolve_file(DATA_DIR, "psd"), use_smote=True, average='weighted')


# ===================================================================
# Everything below matches the ORIGINAL notebook's file layout
# (L+H / L / H bands, M = 1024 / 2048 / 4096, entire signal vs.
# segmented by 10). These files are not in your current "data" folder
# — add them under DATA_DIR (or edit the path) and uncomment the
# matching block to reproduce that part of the notebook.
# ===================================================================

# --- L+H bands, entire signal ---
# train_model(DATA_DIR / "L+H" / "M=1024" / "data.csv", use_smote=True, average='weighted')
# train_model(DATA_DIR / "L+H" / "M=2048" / "data.csv", use_smote=True, average='weighted')
# train_model(DATA_DIR / "L+H" / "M=4096" / "data.csv", use_smote=True, average='weighted')

# --- L+H bands, segmented by factor of 10 ---
# train_model(DATA_DIR / "L+H_seg10" / "M=1024" / "data.csv", use_smote=True, average='weighted')
# train_model(DATA_DIR / "L+H_seg10" / "M=2048" / "data.csv", use_smote=True, average='weighted')
# train_model(DATA_DIR / "L+H_seg10" / "M=4096" / "data.csv", use_smote=True, average='weighted')

# --- L band, entire signal ---
# train_model(DATA_DIR / "L" / "M=1024" / "data.csv", use_smote=True, average='weighted')
# train_model(DATA_DIR / "L" / "M=2048" / "data.csv", use_smote=True, average='weighted')
# train_model(DATA_DIR / "L" / "M=4096" / "data.csv", use_smote=True, average='weighted')

# --- L band, segmented by factor of 10 ---
# train_model(DATA_DIR / "L_seg10" / "M=1024" / "data.csv", use_smote=True, average='weighted')
# train_model(DATA_DIR / "L_seg10" / "M=2048" / "data.csv", use_smote=True, average='weighted')
# train_model(DATA_DIR / "L_seg10" / "M=4096" / "data.csv", use_smote=True, average='weighted')

# --- H band, entire signal ---
# train_model(DATA_DIR / "H" / "M=1024" / "data.csv", use_smote=True, average='weighted')
# train_model(DATA_DIR / "H" / "M=2048" / "data.csv", use_smote=True, average='weighted')
# train_model(DATA_DIR / "H" / "M=4096" / "data.csv", use_smote=True, average='weighted')

# --- H band, segmented by factor of 10 ---
# train_model(DATA_DIR / "H_seg10" / "M=1024" / "data.csv", use_smote=True, average='weighted')
# train_model(DATA_DIR / "H_seg10" / "M=2048" / "data.csv", use_smote=True, average='weighted')
# train_model(DATA_DIR / "H_seg10" / "M=4096" / "data.csv", use_smote=True, average='weighted')
