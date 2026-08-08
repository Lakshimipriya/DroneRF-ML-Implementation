"""
run_dft_features.py
--------------------
Local (VSCode) version of Complete_analysis_on_DFT_extracted_features.ipynb.

Same core train_model() / train_model_for_a_class() as the notebook
(imported unchanged from common_train.py). Colab drive-mount / pip-install
cells removed; paths point at your local data folder.

One-time setup in a terminal:
    pip install xgboost imbalanced-learn scikit-learn numpy pandas

Run:
    python run_dft_features.py
"""

from pathlib import Path
from common_train import train_model

DATA_DIR = Path.home() / "Downloads" / "data dronerf" / "data"

# None of the DFT/FFT band-and-M-value files from the original notebook
# (M=1024/2048/4096 x Both/Upper/Lower bands x segmented/entire) are in
# your current "data" folder. Add them under DATA_DIR using the same
# layout as below, then uncomment the calls you want to run.

# M = 1024, Both bands
# train_model(DATA_DIR / "fft" / "M=1024" / "Both bands" / "L=1e6" / "data.csv", use_smote=True, average='weighted')
# train_model(DATA_DIR / "fft" / "M=1024" / "Both bands" / "entire signal" / "data.csv", use_smote=True, average='weighted')

# M = 1024, Upper band
# train_model(DATA_DIR / "fft" / "M=1024" / "Upper band" / "L=1e6" / "data.csv", use_smote=True, average='weighted')
# train_model(DATA_DIR / "fft" / "M=1024" / "Upper band" / "entire signal" / "data.csv", use_smote=True, average='weighted')

# M = 1024, Lower band
# train_model(DATA_DIR / "fft" / "M=1024" / "Lower band" / "L=1e6" / "data.csv", use_smote=True, average='weighted')
# train_model(DATA_DIR / "fft" / "M=1024" / "Lower band" / "entire signal" / "data.csv", use_smote=True, average='weighted')

# M = 2048, Both bands
# train_model(DATA_DIR / "fft" / "M=2048" / "Both bands" / "L=1e6" / "data.csv", use_smote=True, average='weighted')
# train_model(DATA_DIR / "fft" / "M=2048" / "Both bands" / "entire signal" / "data.csv", use_smote=True, average='weighted')

# M = 2048, Upper band
# train_model(DATA_DIR / "fft" / "M=2048" / "Upper band" / "L=1e6" / "data.csv", use_smote=True, average='weighted')
# train_model(DATA_DIR / "fft" / "M=2048" / "Upper band" / "entire signal" / "data.csv", use_smote=True, average='weighted')

# M = 2048, Lower band
# train_model(DATA_DIR / "fft" / "M=2048" / "Lower band" / "L=1e6" / "data.csv", use_smote=True, average='weighted')
# train_model(DATA_DIR / "fft" / "M=2048" / "Lower band" / "entire signal" / "data.csv", use_smote=True, average='weighted')

# M = 4096, Both bands
# train_model(DATA_DIR / "fft" / "M=4096" / "Both bands" / "L=1e6" / "data.csv", use_smote=True, average='weighted')
# train_model(DATA_DIR / "fft" / "M=4096" / "Both bands" / "entire signal" / "data.csv", use_smote=True, average='weighted')

# M = 4096, Upper band
# train_model(DATA_DIR / "fft" / "M=4096" / "Upper band" / "L=1e6" / "data.csv", use_smote=True, average='weighted')
# train_model(DATA_DIR / "fft" / "M=4096" / "Upper band" / "entire signal" / "data.csv", use_smote=True, average='weighted')

# M = 4096, Lower band
# train_model(DATA_DIR / "fft" / "M=4096" / "Lower band" / "L=1e6" / "data.csv", use_smote=True, average='weighted')
# train_model(DATA_DIR / "fft" / "M=4096" / "Lower band" / "entire signal" / "data.csv", use_smote=True, average='weighted')

if __name__ == "__main__":
    print(
        "No DFT/FFT feature files were found in your data folder yet.\n"
        "Add them under:", DATA_DIR,
        "\nusing the folder layout shown in the comments above, then uncomment "
        "the matching train_model(...) calls."
    )
