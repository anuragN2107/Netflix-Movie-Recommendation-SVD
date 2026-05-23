# Netflix Movie Recommendation System Using SVD

<p align="center">
  <img src="https://images.unsplash.com/photo-1574375927938-d5a98e8edd86?q=80&w=1200&auto=format&fit=crop" alt="Netflix Streaming Interface" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Platform-Google%20Colab-orange?style=flat-square&logo=googlecolab" alt="Google Colab">
  <img src="https://img.shields.io/badge/Language-Python-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Algorithm-SVD-red?style=flat-square" alt="SVD">
</p>

An end-to-end Movie Recommendation Engine built from the ground up, utilizing collaborative filtering and Singular Value Decomposition (SVD) to predict user choices and optimize platform interaction.

---

## 📌 Project Overview
With the immense growth of streaming services, navigating through vast content libraries can be overwhelming for users. This project implements a scalable machine learning recommendation system that uncovers hidden relationships between user historical ratings and movie features to serve personalized content.

* **Course Certification:** Executive PG Certification in Data Science and Artificial Intelligence
* **Organization:** Intellipaat (Associated with iHub Divyasampark, IIT Roorkee)
* **Developed By:** Anurag Srivastva
* **Development Environment:** Google Colab

---

## 🛠️ Tech Stack & Concepts
* **Languages & Environments:** Python, Google Colab
* **Key Methodology:** Collaborative Filtering, Matrix Factorization
* **Algorithm:** Singular Value Decomposition (SVD)
* **Evaluation Metrics:** Root Mean Squared Error (RMSE), Mean Absolute Error (MAE)

---

## 📊 Dataset Description
The project leverages a large-scale dataset comprised of two key components:
1. **Ratings Dataset (`combined.txt`):** Over 24,058,263 rows containing `Customer ID`, `Ratings` (1-5 stars), and `Movie ID`.
2. **Movie Metadata (`movie_titles.csv`):** 17,770 rows detailing `Movie ID`, `Movie Name`, and `Year of Release`.

### Data Preprocessing & Thresholds
To ensure computational feasibility and eliminate extreme sparsity, strict filtering limits were introduced:
* **Minimum threshold per movie:** 1,798 ratings.
* **Minimum threshold per customer:** 52 movie ratings.
* Final processed subset resulted in **4,499 unique movies** and **470,758 unique customers**.

---

## 🧮 How it Works: SVD Architecture
The system builds a highly sparse **User-Item Matrix** ($R$) which is factorized using SVD mathematically represented as:

$$R \approx U \Sigma V^T$$

Where:
* $U$ represents the **User Latent Feature matrix** ($m \times k$).
* $\Sigma$ is a diagonal matrix mapping the **strength of each latent feature** ($k \times k$).
* $V^T$ represents the **Item Latent Feature matrix** ($k \times n$).

Missing movie ratings are predicted via the dot product of these lower-dimensional feature vectors:
$$\hat{R}_{ui} = U_{u} \cdot \Sigma \cdot V_{i}^{T}$$

---

## 📈 Model Performance & Evaluation
Using a **4-fold cross-validation** scheme, the SVD model achieved remarkably low error metrics, showcasing high stability and accuracy:

| Metric | Value Range |
| :--- | :--- |
| **RMSE** | 0.988 - 1.004 |
| **MAE** | 0.786 - 0.805 |

### Key Takeaways
* **Efficiency:** Training time was highly optimized and computationally practical for a 24M dataset.
* **Scalability:** The framework seamlessly captures high-dimensional interactions into lightweight vectors.

---

## 🚀 Quick Links
* 📓 **Notebook:** [View Google Colab Notebook](https://colab.research.google.com/drive/1bgVrGUaPsAbgMmGfxV_QMw9bnZghjUd6#scrollTo=fqVaQ7biGg7e)
* 📄 **Presentation Document:** [Project PDF Details](https://github.com/anuragN2107/Netflix-Movie-Recommendation-SVD/blob/main/Netflix-Recommendation-Engine-Project_Report.docx)
