# 🍿 Netflix Movie Recommendation Engine (SVD Matrix Factorization)

An end-to-end collaborative filtering recommendation system built using **scikit-surprise**, serialized into memory-optimized inference artifacts, and deployed with a responsive **Gradio** user interface on **Hugging Face Spaces**.
===

![Python](https://img.shields.io/badge/Python-3.8+-E50914?style=for-the-badge&logo=python&logoColor=white)
![Framework](https://img.shields.io/badge/Algorithm-SVD%20%2F%20Matrix%20Factorization-black?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Metrics-RMSE%20%2F%20MAE-E50914?style=for-the-badge) 

🚀 **Live Interactive Web App:** [Interact with the Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/anuragN2107/netflix-movie-recommender)

---

## 💼 The Business & Engineering Problem
Predicting customer preferences is central to streaming platform user engagement. However, processing a historical ratings matrix with over 24 million entries presents severe engineering challenges:
1. **The Curse of Sparsity:** Most users rate only a fraction of an enterprise-sized library, leaving vast missing values that traditional distance metrics (like Cosine Similarity) struggle to model meaningfully.
2. **Compute Realities & Latency:** Real-time generation of user pivots over millions of rows easily crashes basic cloud hosting hardware.

### 🧠 The Solution: Latent Factor Modeling via SVD
This project implements **Singular Value Decomposition (SVD)** to handle data sparsity. Instead of storing explicit sparse matrices, the algorithm factorizes the consumer-item interaction framework down into a set of dense latent vectors—uncovering hidden categorical relationships (e.g., genre affinities, thematic links) between user behaviors and content structures.

---

## 🛠️ Core Tech Stack & Tools

| Technology | Purpose | Why We Used It? |
| :--- | :--- | :--- |
| **scikit-surprise** | Core Recommendation Engine | Optimized implementation of Singular Value Decomposition (SVD) with native matrix evaluation metrics. |
| **NumPy & Pandas** | High-performance Data Manipulation | Used for data parsing, restructuring indexing sequences, and removing low-frequency sparse records. |
| **Pickle** | Model Serialization / Asset Packaging | Packs trained matrix factor weights down into sub-50MB binaries, cutting down cloud memory overhead. |
| **Gradio** | UI Layout Engine & Server Wrapper | Creates a web application dashboard with input components, built entirely in Python. |
| **Hugging Face Spaces** | Cloud MLOps Infrastructure Host | Deploys containerized python applications running isolated microservices. |

---

## 📊 Pipeline Architecture
### Preprocessing & Sparsity Reduction
* **Threshold Paring:** To guarantee robust vector convergence, users who rated fewer than 52 movies and items with fewer than 1,799 total ratings were dynamically pruned from the target dataset.
* **Asset Decoupling:** The final trained factor space arrays were extracted and decoupled from the active training database. This allows the lightweight cloud backend script to score and return recommendations instantly without holding massive tables in active RAM.

---

## 🚀 Future Performance Enhancements
* **Hybrid Structural Architecture:** Introducing Content-Based Filtering modules (utilizing text embedding vector representations of synopses) to solve the **Cold-Start Problem** for brand new users.
* **Deep Latent Embeddings (Neural CF):** Replacing standard SVD matrix factorization layers with custom PyTorch Multi-Layer Perceptrons (MLPs) to map non-linear item-user feature correlations.
