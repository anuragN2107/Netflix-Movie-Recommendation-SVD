# 📊 Project Dataset & Source Files

Due to GitHub's file size limits (100 MB), the large-scale Netflix dataset used in this project cannot be hosted directly in this repository. 

You can access and download the complete raw dataset here:
👉 **[Download Dataset from Google Drive](PASTE_YOUR_GOOGLE_DRIVE_LINK_HERE)**

---

## 📁 Dataset Architecture

The project relies on two primary data files to train and evaluate the Singular Value Decomposition (SVD) recommendation engine:

### 1. `combined.txt` (User Ratings Dataset)
* [cite_start]**Description:** Contains raw, historical user interactions and ratings[cite: 65, 81].
* [cite_start]**File Size:** ~24,058,263 rows $\times$ 3 columns[cite: 71].
* **Feature Columns:**
  * [cite_start]`Customer ID`: Unique identifier for each customer[cite: 68].
  * [cite_start]`Ratings`: Rating score given by the customer (1 to 5 stars)[cite: 69].
  * [cite_start]`Movie ID`: Unique identifier for each movie[cite: 70].

### 2. `movie_titles.csv` (Movie Metadata)
* [cite_start]**Description:** Provides metadata to map unique Movie IDs to their structural information[cite: 72, 81].
* [cite_start]**File Size:** ~17,770 rows $\times$ 3 columns[cite: 80].
* **Feature Columns:**
  * [cite_start]`Movie ID`: Unique identifier for each movie[cite: 77].
  * [cite_start]`Movie Name`: Title of the movie[cite: 78].
  * [cite_start]`Year of Release`: The calendar year the movie was released[cite: 79].

---

## 🧹 Data Preprocessing & Filtering Rules

The raw matrix is highly sparse. [cite_start]To build a reliable, computationally feasible collaborative filtering model, the following data thresholds were strictly enforced during preprocessing[cite: 58, 82, 92, 95]:

* [cite_start]**Movie Popularity Cutoff:** A movie must have received at least **1,798 ratings** to be included[cite: 93].
* [cite_start]**User Engagement Cutoff:** A customer must have rated at least **52 movies** to be included[cite: 94].

### Final Processed Dataset Insights
[cite_start]After applying the custom loop logic and cutoff criteria, the modeled subset contains[cite: 89, 95]:
* [cite_start]**Total Unique Movies:** 4,499 [cite: 99]
* [cite_start]**Total Unique Customers:** 470,758 [cite: 100]
* [cite_start]**Total Modeled Ratings:** 24,053,764 [cite: 102]
* [cite_start]**Most Frequent Rating:** 4 Stars [cite: 101]
