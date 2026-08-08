"""
run_time_domain_features.py
----------------------------
Local (VSCode) version of Time_domain_features.ipynb.

Same core train_model() / train_model_for_a_class() as the notebook
(imported unchanged from common_train.py). Colab drive-mount / pip-install
cells removed; the ad-hoc "Some testing" cells at the end of the
notebook (which referenced an undefined `Data` variable and duplicated
the cross-validation loop by hand) were left out since they weren't
part of the reusable train_model() pipeline and would error as-is.

One-time setup in a terminal:
    pip install xgboost imbalanced-learn scikit-learn numpy pandas

Run:
    python run_time_domain_features.py
"""

from pathlib import Path
from common_train import train_model, resolve_file

DATA_DIR = Path.home() / "Downloads" / "data dronerf" / "data"


# ===================================================================
# Files that ARE currently in your data folder
# ===================================================================

print(">>> RMS energy, window length = 512")
train_model(resolve_file(DATA_DIR, "rms_energy_wl=512"), use_smote=True, average='weighted')

print(">>> RMS energy, window length = 2048")
train_model(resolve_file(DATA_DIR, "rms_energy_wl=2048"), use_smote=True, average='weighted')

print(">>> Zero crossing rate, window length = 512")
train_model(resolve_file(DATA_DIR, "zero_crossing_rate_wl=512"), use_smote=True, average='weighted')

print(">>> Spectral centroid, window length = 512")
train_model(resolve_file(DATA_DIR, "spectral_centroid_512"), use_smote=True, average='weighted')

print(">>> Spectral centroid, window length = 2048")
train_model(resolve_file(DATA_DIR, "spectral_centroid_2048"), use_smote=True, average='weighted')


# ===================================================================
# From the original notebook, not currently in your data folder:
# RMS energy / zero-crossing-rate at window lengths 1e6 and 1e7.
# Add the files under DATA_DIR and uncomment to run them.
# ===================================================================

# train_model(resolve_file(DATA_DIR, "rms_energy_M=1e6"), use_smote=True, average='weighted')
# train_model(resolve_file(DATA_DIR, "rms_energy_M=1e7"), use_smote=True, average='weighted')
# train_model(resolve_file(DATA_DIR, "zero_crossing_rate_1e6"), use_smote=True, average='weighted')
# train_model(resolve_file(DATA_DIR, "zero_crossing_rate_1e7"), use_smote=True, average='weighted')
