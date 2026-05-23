# 🎬 Netflix Movie Recommendation Engine (SVD)
===

![Python](https://img.shields.io/badge/Python-3.8+-E50914?style=for-the-badge&logo=python&logoColor=white)
![Framework](https://img.shields.io/badge/Algorithm-SVD%20%2F%20Matrix%20Factorization-black?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Metrics-RMSE%20%2F%20MAE-E50914?style=for-the-badge)

> **System Status:** Recommendation matrix fully optimized. This repository contains an end-to-end Machine Learning pipeline utilizing Singular Value Decomposition (SVD) to combat the data sparsity problem and predict highly accurate user-movie rating vectors.

---

## 🚀 Live Environment Hub

| Production Asset | Interface Access Link | Deployment Status |
| :--- | :--- | :--- |
| **🤖 Model Engine** | (https://github.com/anuragN2107/Netflix-Movie-Recommendation-SVD/blob/main/Capstone_Project(Netflix).ipynb) | `● OPERATIONAL` |
| **📑 Technical Brief** | (https://github.com/anuragN2107/Netflix-Movie-Recommendation-SVD/blob/main/Netflix-Recommendation-Engine-Project_Report.docx) | `● VERIFIED` |
| **🗃️ Training Matrix** | (https://drive.google.com/drive/folders/1T4kugstkoC8PJIRq9FCtIpHdlySXjpvG) | `● SECURE` |

---

## 🧠 Architectural Overview

### 🍿 1. Matrix Factorization Pipeline
* **Dimensionality Reduction:** Resolved high-dimensional user-item interaction matrices into low-rank dense latent spaces using **Singular Value Decomposition (SVD)**.
* **Sparsity Mitigation:** Addressed the massive missing-rating issue typical of large-scale streaming catalogs by capturing underlying user preferences and movie features.

### 📐 2. Algorithmic Optimization
* **Mathematical Optimization:** Implemented bias terms to account for individual user rating tendencies (e.g., critical vs. lenient raters) and inherent movie popularity shifts.
* **Hyperparameter Tuning:** Fine-tuned the learning rate ($\gamma$) and regularization parameter ($\lambda$) to optimize performance metrics and prevent model overfitting.

---

## 📊 Model Evaluation Summary

| Algorithm Component | Primary Metric Target | Optimization Strategy |
| :--- | :--- | :--- |
| **SVD Baseline** | Minimize Root Mean Squared Error (RMSE) | Stochastic Gradient Descent (SGD) |
| **Latent Factors ($k$)** | Capture Implicit User Preferences | Matrix Rank Decomposition |
| **Regularization ($\lambda$)** | Control Overfitting on Sparse Data | L2 Ridge Penalization |

---

## 🛠️ Data Science Stack
* **Language Environment:** Python 3.8+ 🐍
* **Recommendation Framework:** Surprise Scikit / Custom SVD Engines
* **Data Manipulation:** Pandas, NumPy
* **Visualization Matrix:** Seaborn Cinematic Palettes, Matplotlib

---
*Developed as part of an advanced machine learning portfolio framework.*
