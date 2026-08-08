# DroneRF ML Implementation

A **local implementation and study** of machine-learning techniques for **RF-based drone detection and identification** using the **DroneRF dataset**.

This project replicates an existing research framework and adapts its machine-learning pipeline from **Google Colab to Visual Studio Code** for local execution. It focuses on understanding and evaluating different RF feature representations using **XGBoost**.

### Features Studied

* Power Spectral Density (PSD)
* MFCC
* RMS Energy
* Spectral Centroid
* Zero-Crossing Rate
* Feature Fusion

### Classification Tasks

* **2-Class:** Drone vs Background
* **4-Class:** Drone Type Classification
* **10-Class:** Detailed Drone/Activity Identification

### Tools & Technologies

* Python
* Visual Studio Code
* MATLAB
* XGBoost
* Scikit-learn
* Imbalanced-learn
* DroneRF Dataset
    data drive - https://drive.google.com/drive/folders/1wJPr_D0I53Ac_3XrsK1TcGjxTm7QWX2n?usp=sharing

### Methodology

**RF Signal → Feature Extraction → Normalization → XGBoost → 5-Fold Stratified Cross-Validation → Performance Evaluation**

### Note

This repository is **for educational and research purposes**. It is an implementation/replication study and does **not** introduce a new algorithm, dataset, or hardware system.

### Author

**Lakshimi Priya V**
Electronics and Communication Engineering
Alagappa Chettiar Government College of Engineering & Technology, Karaikudi
BSERC - Def space summer internship project 
