# Netflix Movie Recommendation System Using SVD

<p align="center">
  <img src="https://images.unsplash.com/photo-1574375927938-d5a98e8edd86?q=80&w=1200&auto=format&fit=crop" alt="Netflix Streaming Interface" width="100%">
</p>

[cite_start]An end-to-end Movie Recommendation Engine built from the ground up, utilizing collaborative filtering and Singular Value Decomposition (SVD) to predict user choices and optimize platform interaction[cite: 7, 27, 43].

---

## 📌 Project Overview
[cite_start]With the immense growth of streaming services, navigating through vast content libraries can be overwhelming for users[cite: 52, 53]. [cite_start]This project implements a scalable machine learning recommendation system that uncovers hidden relationships between user historical ratings and movie features to serve personalized content[cite: 54, 55, 59].

* [cite_start]**Course Certification:** Executive PG Certification in Data Science and Artificial Intelligence [cite: 3]
* [cite_start]**Organization:** Intellipaat (Associated with iHub Divyasampark, IIT Roorkee) [cite: 4]
* [cite_start]**Developed By:** Anurag Srivastva [cite: 2]
* [cite_start]**Development Environment:** Google Colab [cite: 3]

---

## 🛠️ Tech Stack & Concepts
* [cite_start]**Languages & Environments:** Python, Google Colab [cite: 3]
* [cite_start]**Key Methodology:** Collaborative Filtering, Matrix Factorization [cite: 23, 27]
* [cite_start]**Algorithm:** Singular Value Decomposition (SVD) [cite: 7, 9]
* [cite_start]**Evaluation Metrics:** Root Mean Squared Error (RMSE), Mean Absolute Error (MAE) [cite: 26, 179]

---

## 📊 Dataset Description
[cite_start]The project leverages a large-scale dataset comprised of two key components[cite: 64]:
1. [cite_start]**Ratings Dataset (`combined.txt`):** Over 24,058,263 rows containing `Customer ID`, `Ratings` (1-5 stars), and `Movie ID`[cite: 65, 68, 69, 70, 71].
2. [cite_start]**Movie Metadata (`movie_titles.csv`):** 17,770 rows detailing `Movie ID`, `Movie Name`, and `Year of Release`[cite: 49, 77, 78, 79, 80].

### Data Preprocessing & Thresholds
[cite_start]To ensure computational feasibility and eliminate extreme sparsity, strict filtering limits were introduced[cite: 82, 92, 95, 171, 172]:
* [cite_start]**Minimum threshold per movie:** 1,798 ratings[cite: 93].
* **Minimum threshold per customer:** 52 movie ratings[cite: 94].
* [cite_start]Final processed subset resulted in **4,499 unique movies** and **470,758 unique customers**[cite: 99, 100].

---

## 🧮 How it Works: SVD Architecture
[cite_start]The system builds a highly sparse **User-Item Matrix** ($R$) which is factorized using SVD mathematically represented as[cite: 111, 129, 130, 158]:

$$R \approx U \Sigma V^T$$

Where:
* $U$ represents the **User Latent Feature matrix** ($m \times k$)[cite: 132].
* [cite_start]$\Sigma$ is a diagonal matrix mapping the **strength of each latent feature** ($k \times k$)[cite: 133].
* [cite_start]$V^T$ represents the **Item Latent Feature matrix** ($k \times n$)[cite: 134].

[cite_start]Missing movie ratings are predicted via the dot product of these lower-dimensional feature vectors[cite: 127, 139, 140, 151]:
$$\hat{R}_{ui} = U_{u} \cdot \Sigma \cdot V_{i}^{T}$$

---

## 📈 Model Performance & Evaluation
[cite_start]Using a **4-fold cross-validation** scheme, the SVD model achieved remarkably low error metrics, showcasing high stability and accuracy[cite: 180, 184, 187]:

| Metric | Value Range |
| :--- | :--- |
| **RMSE** | [cite_start]0.988 - 1.004 [cite: 182] |
| **MAE** | [cite_start]0.786 - 0.805 [cite: 183] |

### Key Takeaways
* [cite_start]**Efficiency:** Training time was highly optimized and computationally practical for a 24M dataset[cite: 165, 188].
* **Scalability:** The framework seamlessly captures high-dimensional interactions into lightweight vectors[cite: 153, 160].

---

## 🚀 Quick Links
* [cite_start]📓 **Notebook:** [View Google Colab Notebook](https://colab.research.google.com/drive/1bgVrGUaPsAbgMmGfxV_QMw9bnZghjUd6#scrollTo=fqVaQ7biGg7e) [cite: 225]
* 📄 **Presentation Document:** [Project PDF Details](PASTE_YOUR_FULL_GITHUB_URL_HERE)
