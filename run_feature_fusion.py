"""
run_feature_fusion.py
----------------------
Local (VSCode) version of Feature_Fusion_experiments.ipynb.

Same core train_model() / train_model_for_a_class() as the notebook
(imported unchanged from common_train.py). Colab drive-mount / pip-install
cells removed; paths point at your local data folder.

One-time setup in a terminal:
    pip install xgboost imbalanced-learn scikit-learn numpy pandas

Run:
    python run_feature_fusion.py
"""

from pathlib import Path
from common_train import train_model

DATA_DIR = Path.home() / "Downloads" / "data dronerf" / "data"

# None of the feature-fusion CSVs from the original notebook are in your
# current "data" folder. Put them under DATA_DIR (any subfolder layout
# you like, just update the paths below) and uncomment what you need.

# Parameter setting 1
# train_model(DATA_DIR / "feature_fusion_2048.csv", use_smote=True, average='weighted')

# Parameter setting 2
# train_model(DATA_DIR / "feature_fusion_4096.csv", use_smote=True, average='weighted')

# Fuse MFCC and FFT features
# train_model(DATA_DIR / "feature_fusion_mfcc_fft.csv", use_smote=True, average='weighted')

# Fuse MFCC and PSD features
# train_model(DATA_DIR / "feature_fusion_mfcc_psd.csv", use_smote=True, average='weighted')

# Fuse FFT and PSD features
# train_model(DATA_DIR / "feature_fusion_fft_psd.csv", use_smote=True, average='weighted')

if __name__ == "__main__":
    print(
        "No feature-fusion CSVs were found in your data folder yet.\n"
        "Add them under:", DATA_DIR,
        "\nand uncomment the matching train_model(...) calls above."
    )
