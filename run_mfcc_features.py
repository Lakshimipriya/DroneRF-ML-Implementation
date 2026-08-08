"""
run_mfcc_features.py
---------------------
Local (VSCode) version of Complete_analysis_on_MFCC_features.ipynb.

This notebook was written differently from the other four (PSD, DFT,
Time-domain, Fusion): instead of calling a shared train_model()
function, each section manually slices the data and runs its own
StratifiedKFold + XGBClassifier / SVC loop. That per-section logic is
kept EXACTLY as it was in the notebook.

The one necessary change: the notebook hardcoded the feature/label
column split for each specific file (e.g. `Data[0:136668,:]`,
`Data[0:28,:]`), which only works for the exact original Google-Drive
files. Since your local files may not be shaped the same way, the
split is now computed automatically the same way common_train.py does
it (last 3 rows = the three label sets, everything above = features).
The classification logic itself (XGBClassifier/SVC, StratifiedKFold,
scoring) is untouched.

One-time setup in a terminal:
    pip install xgboost scikit-learn numpy pandas

Run:
    python run_mfcc_features.py
"""

from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier

from common_train import resolve_file

DATA_DIR = Path.home() / "Downloads" / "data dronerf" / "data"


def load_features_and_labels(path):
    """Same row layout used throughout the notebooks: the last 3 rows
    are Label_1 (2-class), Label_2 (4-class), Label_3 (10-class); every
    row above that is a feature."""
    Data = np.loadtxt(path, delimiter=",")
    print(Data.shape)
    num_features = Data.shape[0] - 3
    x = np.transpose(Data[0:num_features, :])
    Label_1 = np.transpose(Data[num_features:num_features + 1, :]).astype(int)
    Label_2 = np.transpose(Data[num_features + 1:num_features + 2, :]).astype(int)
    Label_3 = np.transpose(Data[num_features + 2:num_features + 3, :]).astype(int)
    return x, Label_1, Label_2, Label_3


def run_xgb_cv(x, Label, K=5, random_state=1, label_name=""):
    cvscores = []
    cnt = 0
    kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=random_state)
    for train, test in kfold.split(x, Label):
        cnt += 1
        model = XGBClassifier()
        model.fit(x[train], Label[train])
        score = model.score(x[test], Label[test])
        print("Cross validation {} : {}".format(cnt, score))
        cvscores.append(score)
    print(f"Mean accuracy ({label_name}): {np.mean(cvscores)}\n")
    return cvscores


def run_svc_cv(x, Label, K=5, random_state=1, label_name=""):
    cvscores = []
    cnt = 0
    kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=random_state)
    for train, test in kfold.split(x, Label):
        cnt += 1
        model = SVC()
        model.fit(x[train], Label[train])
        score = model.score(x[test], Label[test])
        print("Cross validation {} : {}".format(cnt, score))
        cvscores.append(score)
    print(f"Mean accuracy ({label_name}): {np.mean(cvscores)}\n")
    return cvscores


# ===================================================================
# MFCC features (matches "Copy of mfcc_features" in your data folder)
# ===================================================================
print(">>> MFCC features")
path = resolve_file(DATA_DIR, "mfcc_features")
x, Label_1, Label_2, Label_3 = load_features_and_labels(path)

print("XGBoost - 2 class")
run_xgb_cv(x, Label_1, label_name="2 class")
print("XGBoost - 4 class")
run_xgb_cv(x, Label_2, label_name="4 class")
print("XGBoost - 10 class")
run_xgb_cv(x, Label_3, label_name="10 class")

print("SVM - 2 class")
run_svc_cv(x, Label_1, label_name="2 class")
print("SVM - 4 class")
run_svc_cv(x, Label_2, label_name="4 class")
print("SVM - 10 class")
run_svc_cv(x, Label_3, label_name="10 class")


# ===================================================================
# MFCC features, W = 2048 (matches "Copy of mfcc_features_2048")
# ===================================================================
print(">>> MFCC features, W = 2048")
path = resolve_file(DATA_DIR, "mfcc_features_2048")
x, Label_1, Label_2, Label_3 = load_features_and_labels(path)

print("XGBoost - 2 class")
run_xgb_cv(x, Label_1, label_name="2 class")
print("XGBoost - 4 class")
run_xgb_cv(x, Label_2, label_name="4 class")
print("XGBoost - 10 class")
run_xgb_cv(x, Label_3, label_name="10 class")


# ===================================================================
# PSD features - segmented, both bands, M=2048, L=1e5
# (matches "Copy of psd_both_bands_M=2048_L=1e5")
# ===================================================================
print(">>> PSD features - both bands, M=2048, L=1e5")
path = resolve_file(DATA_DIR, "psd_both_bands")
x, Label_1, Label_2, Label_3 = load_features_and_labels(path)

print("XGBoost - 2 class")
run_xgb_cv(x, Label_1, label_name="2 class")
print("XGBoost - 4 class")
run_xgb_cv(x, Label_2, label_name="4 class")
print("XGBoost - 10 class")
run_xgb_cv(x, Label_3, label_name="10 class")


# ===================================================================
# PSD features - treat entire input signal as whole
# (matches "Copy of psd")
# ===================================================================
print(">>> PSD features - whole signal")
path = resolve_file(DATA_DIR, "psd")
x, Label_1, Label_2, Label_3 = load_features_and_labels(path)

print("XGBoost - 2 class")
run_xgb_cv(x, Label_1, label_name="2 class")
print("XGBoost - 4 class")
run_xgb_cv(x, Label_2, label_name="4 class")
print("XGBoost - 10 class")
run_xgb_cv(x, Label_3, label_name="10 class")


# ===================================================================
# From the original notebook, not currently in your data folder:
# MFCC (M0E, frame length = 4096) and MFCC voicebox W=4096.
# Add the files under DATA_DIR and uncomment to run them.
# ===================================================================

# print(">>> MFCC M0E, frame length = 4096")
# path = resolve_file(DATA_DIR, "mfcc_features_M0E_frame_len")
# x, Label_1, Label_2, Label_3 = load_features_and_labels(path)
# run_xgb_cv(x, Label_1, label_name="2 class")
# run_xgb_cv(x, Label_2, label_name="4 class")
# run_xgb_cv(x, Label_3, label_name="10 class")

# print(">>> MFCC voicebox, W = 4096")
# path = resolve_file(DATA_DIR, "mfcc_features_voicebox_W")
# x, Label_1, Label_2, Label_3 = load_features_and_labels(path)
# run_xgb_cv(x, Label_1, label_name="2 class")
# run_xgb_cv(x, Label_2, label_name="4 class")
# run_xgb_cv(x, Label_3, label_name="10 class")
