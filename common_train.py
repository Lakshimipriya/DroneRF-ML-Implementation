"""
common_train.py
----------------
Shared code used by every analysis script (PSD, DFT, MFCC, Time-domain,
Feature Fusion). This is the SAME core logic that was in the original
Colab notebooks (train_model_for_a_class / train_model) — nothing about
the machine-learning logic has been changed. The only thing added is
`resolve_file()`, a small helper so the scripts can find your local
files (e.g. "Copy of psd") without you having to know the exact
extension/prefix.

Install requirements once, in a terminal (not inside the script):
    pip install xgboost imbalanced-learn scikit-learn numpy pandas
"""

import os
import warnings

# Basic data handling libraries
import numpy as np
import pandas as pd
np.random.seed(42)

# Cross validation and hyperparameter tuning libraries
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Machine learning classifiers
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')


class color:
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


smote = SMOTE()


# ---------------------------------------------------------------------
# CORE FUNCTIONS — identical logic to the original notebooks
# ---------------------------------------------------------------------

def train_model_for_a_class(x, Label, K, use_smote, average='binary'):
    # Cross validation and model training
    accuracy_scores = []
    f1_scores = []
    recall_scores = []
    precision_scores = []
    kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

    for train, test in kfold.split(x, Label):
        if len(np.unique(Label)) == 2:
            model = XGBClassifier(objective="binary:logistic", seed=42)
        else:
            model = XGBClassifier(objective="multi:softmax", seed=42)
        if use_smote:
            X_train_smote, y_train_smote = smote.fit_resample(x[train], Label[train])
            model.fit(X_train_smote, y_train_smote)
        else:
            model.fit(x[train], Label[train])
        y_pred = model.predict(x[test])
        accuracy_scores.append(accuracy_score(Label[test], y_pred))
        f1_scores.append(f1_score(Label[test], y_pred, average=average))
        recall_scores.append(recall_score(Label[test], y_pred, average=average))
        precision_scores.append(precision_score(Label[test], y_pred, average=average))

    print("Accuracy: {}".format(np.mean(accuracy_scores)))
    print("f1_score: {}".format(np.mean(f1_scores)))
    print("recall_score: {}".format(np.mean(recall_scores)))
    print("precision_score: {}".format(np.mean(precision_scores)))
    print("\n-------------------------------------------------------\n")


def train_model(data_file_path, use_smote=False, average='binary'):

    data = np.loadtxt(data_file_path, delimiter=",")

    num_samples = data.shape[1]
    num_features = data.shape[0] - 3

    x = np.transpose(data[0:num_features:])
    Label_1 = np.transpose(data[num_features:num_features + 1, :]); Label_1 = Label_1.astype(int)
    Label_2 = np.transpose(data[num_features + 1:num_features + 2, :]); Label_2 = Label_2.astype(int)
    Label_3 = np.transpose(data[num_features + 2:num_features + 3, :]); Label_3 = Label_3.astype(int)

    print("Number of points in the dataset: {}".format(num_samples))
    print("Number of features in each datapoint: {}\n".format(num_features))

    # Preprocessing
    scl = StandardScaler()
    x = scl.fit_transform(x)

    print(color.BOLD + "Training model for 2 class:" + color.END)
    train_model_for_a_class(x, Label_1, 5, use_smote)

    print(color.BOLD + "Training model for 4 class:" + color.END)
    train_model_for_a_class(x, Label_2, 5, use_smote, average=average)

    print(color.BOLD + "Training model for 10 class:" + color.END)
    train_model_for_a_class(x, Label_3, 5, use_smote, average=average)


# ---------------------------------------------------------------------
# NEW HELPER (not in the original notebooks) — finds a local file by
# keyword so you don't need to fix "Copy of ..." names / extensions by
# hand. It does NOT touch any of the ML logic above.
# ---------------------------------------------------------------------

def resolve_file(data_dir, keyword):
    """
    Look inside `data_dir` for a file whose (cleaned) name contains
    `keyword`. Ignores case, spaces, '=' signs, and an accidental
    'Copy of ' prefix, so 'Copy of psd_both_bands_M=2048_L=1e5.csv'
    will match keyword='psd_both_bands'.
    """
    data_dir = str(data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data folder not found: {data_dir}\n"
            f"Edit DATA_DIR at the top of the script to point at your "
            f"'data' folder."
        )

    def clean(s):
        stem = os.path.splitext(s)[0]
        return stem.lower().replace("copy of ", "").replace(" ", "").replace("=", "")

    keyword_clean = clean(keyword)
    all_files = os.listdir(data_dir)

    # Prefer an exact stem match (e.g. keyword "psd" matching "psd.csv"
    # rather than "psd_both_bands_M=2048_L=1e5.csv") before falling
    # back to a substring match.
    exact = [f for f in all_files if clean(f) == keyword_clean]
    if exact:
        matches = exact
    else:
        matches = [f for f in all_files if keyword_clean in clean(f)]

    if not matches:
        raise FileNotFoundError(
            f"Could not find a file matching '{keyword}' in {data_dir}\n"
            f"Files available there: {all_files}"
        )
    if len(matches) > 1:
        print(f"[note] multiple files matched '{keyword}': {matches} -> using '{matches[0]}'")

    return os.path.join(data_dir, matches[0])
