project_metadata:
  title: "Python Project - Capstone Project (Netflix)"
  prepared_by: "Anurag Srivastva"
  course: "Executive PG Certification in Data Science and Artificial Intelligence"
  tool_used: "Google Colab"
  organisation: "Intellipaat (Associated with iHub Divyasampark, IIT Roorkee)"

objectives:
  - "Find out the list of most popular and liked genre."
  - "Create a Model that finds the best suited Movie for one user in every genre."
  - "Find what Genre Movies have received the best and worst ratings based on User Rating."

datasets: |
  The datasets utilized for this capstone project can be accessed via the following link:
  - [Google Drive Dataset Folder](https://drive.google.com/drive/folders/1T4kugstkoC8PJIRq9FCtIpHdlySXjpvG?usp=sharing)

introduction: |
  With the rapid growth of digital streaming platforms such as Netflix, Amazon Prime, and Disney+, users are exposed to a vast collection of movies and television content. Navigating through this large volume of content can be overwhelming, making it difficult for users to find movies that match their interests. To address this challenge, Movie Recommendation Systems play a crucial role in delivering personalized content suggestions based on user preferences and behavior.

  A movie recommendation system analyzes historical user interactions such as ratings and viewing patterns to predict movies that a user is likely to enjoy. These systems enhance user experience by reducing search effort, increasing user engagement, and improving content consumption. Among various recommendation techniques, collaborative filtering has proven to be one of the most effective approaches, as it leverages similarities between users and items.

  In this project, a Singular Value Decomposition (SVD) based collaborative filtering approach is implemented to build a scalable and accurate movie recommendation system. SVD efficiently handles large and sparse datasets by extracting latent features that represent hidden relationships between users and movies. The system is trained and evaluated using a large-scale dataset containing millions of movie ratings, demonstrating its effectiveness in real-world recommendation scenarios. This project highlights the importance of data preprocessing, model selection, and evaluation techniques in building a reliable recommendation system and provides insights into how modern streaming platforms utilize machine learning algorithms to deliver personalized user experiences.

data_description:
  ratings_dataset:
    file_name: "combined.txt"
    columns:
      - "Customer ID: Unique identifier for each customer."
      - "Ratings: Ratings given by the customer to a movie (1 to 5 stars)."
      - "Movie ID: Unique identifier for each movie."
    size: "24,058,263 rows × 2 columns"
  movie_metadata:
    file_name: "MOVIE TITLES"
    columns:
      - "Movie ID: Unique identifier for each movie."
      - "Movie Name: Title of the movie."
      - "Year of Release: Release year of the movie."
    size: "17,770 rows × 3 columns"
  notes: "combined.txt contains the raw user ratings, while MOVIE TITLES provides metadata to map Movie IDs to names and release years. This structure is suitable for collaborative filtering and recommendation system analysis."

data_preprocessing:
  steps:
    - "Decoding Data: Data is encoded, so it needs to be decoded first."
    - "Header Allocation: Column names were missing in the raw file, so appropriate names were assigned."
    - "Pattern Observation: It was observed that rows containing only Movie ID have NaN in the Ratings column. Customer ratings appear in the rows following the Movie ID."
    - "Custom Loop Logic: Null values in the Ratings column are used to determine the range of Customer IDs who rated each movie."
    - "Filtering / Cutoff Criteria: Minimum rating threshold for movies: 1798; Minimum number of ratings per customer: 52. Only data meeting these thresholds is included for modeling."
    - "Data Type Adjustments: Converted relevant columns from object to float/int for calculations and modeling."

data_insights:
  total_movies: 4499
  total_customers: 470758
  most_frequent_rating: "4 stars"
  total_ratings: 24053764

svd_mathematical_formulation:
  explanation: "For a given user-item rating matrix R of size m x n (where m is the number of users and n is the number of movies):"
  equation: "R ≈ U × Σ × V^T"
  latent_factors:
    U: "an m x k matrix representing user latent features."
    Sigma: "a k x k diagonal matrix containing singular values, which represent the strength of each latent feature."
    VT: "a k x n matrix representing item latent features."
    k: "the number of latent factors (chosen based on the dataset)."
  intuition:
    text: "Each user and each item is represented as a vector in a k-dimensional latent space. The dot product of the user and item vectors approximates the rating a user would give to an item."
    prediction_equation: "R_hat_ui = U_u · Σ · V_i^T"

model_evaluation:
  metrics:
    - "RMSE values: approximately 0.988 to 1.004"
    - "MAE values: approximately 0.786 to 0.805"
  notes: "4-fold cross-validation was used to assess the model’s performance. These metrics indicate that the SVD algorithm provides reasonably accurate predictions, as the error values are relatively low."

google_colab_link: "https://colab.research.google.com/drive/1bgVrGUaPsAbgMmGfxV_QMw9bnZghjUd6#scrollTo=fqVaQ7biGg7e"
