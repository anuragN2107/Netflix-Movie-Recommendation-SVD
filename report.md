
 Python Project - Capstone Project (Netflix)  

	Prepared By: Anurag Srivastva
	Course: Executive PG Certification in Data Science and Artificial Intelligence
	Tool Used: Google Colab
	Organisation: Intellipaat (Associated with iHub Divyasampark, IIT Roorke)

 





Table of Contents
Section	Topic	Page
		Table of Content	2
		Problem Statement	3
		Objectives	3
		Datasets	3
		Introduction	4
		Data Description	5
		Data Pre-processing Steps and Inspiration
	6
		Data Insights	6
		Movie Recommendation System overview	7
		SVD	8
		Choosing the Algorithm for the Project	9
		Motivation and Reasons for Choosing SVD	10
		Assumptions	11
		Model Evaluation and Techniques	11
		Inferences from Model Evaluation	11
		Conclusion	12
		Future of Netflix Movie Recommendation System	12-13
		Google Colab Link	13

Problem Statement
Customer Behaviour and its prediction lies at the core of every Business Model. From Stock Exchange, e-Commerce and Automobile to even Presidential Elections, predictions serve a great purpose. Most of these predictions are based on the data available about a person’s activity either online or in-person
Recommendation Engines are the much-needed manifestations of the desired Predictability of User Activity. Recommendation Engines move one step further and not only give information but put forth strategies to further increase users’ interaction with the platform.
In today’s world OTT platform and Streaming Services have taken up a big chunk in the Retail and Entertainment industry. Organizations like Netflix, Amazon etc. analyse User Activity Pattern’s and suggest products that better suit the user needs and choices.
For the purpose of this Project we will be creating one such Recommendation Engine from the ground-up, where every single user, based on there area of interest and ratings, would be recommended a list of movies that are best suited for them.
________________________________________

Objectives
1.Find out the list of most popular and liked genre
2.Create Model that finds the best suited Movie for one user in every genre.
3.Find what Genre Movies have received the best and worst ratings based on User Rating.
________________________________________


Datasets

https://drive.google.com/drive/folders/1T4kugstkoC8PJIRq9FCtIpHdlySXjpvG?usp=sharing

________________________________________
Introduction
With the rapid growth of digital streaming platforms such as Netflix, Amazon Prime, and Disney+, users are exposed to a vast collection of movies and television content. Navigating through this large volume of content can be overwhelming, making it difficult for users to find movies that match their interests. To address this challenge, Movie Recommendation Systems play a crucial role in delivering personalized content suggestions based on user preferences and behavior.
A movie recommendation system analyzes historical user interactions such as ratings and viewing patterns to predict movies that a user is likely to enjoy. These systems enhance user experience by reducing search effort, increasing user engagement, and improving content consumption. Among various recommendation techniques, collaborative filtering has proven to be one of the most effective approaches, as it leverages similarities between users and items.
In this project, a Singular Value Decomposition (SVD) based collaborative filtering approach is implemented to build a scalable and accurate movie recommendation system. SVD efficiently handles large and sparse datasets by extracting latent features that represent hidden relationships between users and movies. The system is trained and evaluated using a large-scale dataset containing millions of movie ratings, demonstrating its effectiveness in real-world recommendation scenarios.
This project highlights the importance of data preprocessing, model selection, and evaluation techniques in building a reliable recommendation system and provides insights into how modern streaming platforms utilize machine learning algorithms to deliver personalized user experiences.








________________________________________
Data Description
The project consists of two files:
	combined.txt – Ratings Dataset
	Columns:
	Customer ID: Unique identifier for each customer.
	Ratings: Ratings given by the customer to a movie (1 to 5 stars).
	Movie ID: Unique identifier for each movie.
	Size: 24,058,263 rows × 2 columns
	MOVIE TITLES – Movie Metadata
	Columns:
	Movie ID: Unique identifier for each movie.
	Movie Name: Title of the movie.
	Year of Release: Release year of the movie.
	Size: 17,770 rows × 3 columns
Notes:
	combined.txt contains the raw user ratings, while MOVIE TITLES provides metadata to map Movie IDs to names and release years.
	This structure is suitable for collaborative filtering and recommendation system analysis.






________________________________________

Data Pre-processing Steps and Inspiration
	Data is encoded, so it needs to be decoded first.
	Column names were missing in the raw file, so appropriate names were assigned.
	Observed pattern:
	Rows containing only Movie ID have NaN in the Ratings column.
	Customer ratings appear in the rows following the Movie ID.
	Custom loop logic:
	Null values in the Ratings column are used to determine the range of Customer IDs who rated each movie.
	Filtering / Cutoff Criteria:
	Minimum rating threshold for movies: 1798
	Minimum number of ratings per customer: 52
	Only data meeting these thresholds is included for modeling.
	Data type adjustments:
	Converted relevant columns from object to float/int for calculations and modeling.

________________________________________

Data Insights
	Total movies in the dataset: 4,499
	Total customers: 470,758
	Most frequent movie rating: 4 stars
	Total ratings given by all customers: 24,053,764

________________________________________
Movie Recommendation System Overview
A Movie Recommendation System is a type of recommender system designed to suggest movies to users based on their preferences, viewing history, and the behavior of similar users. Its primary goal is to help users discover movies they are likely to enjoy, while also increasing engagement on streaming platforms like Netflix or Amazon Prime.How it Works
	Data Collection:
	Collect user ratings, watch history, and movie metadata such as genre, release year, and cast.
	Data Preprocessing:
	Organize the data into a user-item interaction matrix, handle missing ratings, and normalize values for analysis.
	Recommendation Algorithms:
	Collaborative Filtering: Suggests movies based on the preferences of users with similar tastes.
	Content-Based Filtering: Suggests movies similar to those a user liked, based on movie features.
	Hybrid Approaches: Combine collaborative and content-based methods for better accuracy.
	Prediction and Recommendation:
	Algorithms like SVD (Singular Value Decomposition) factorize the user-item matrix to learn latent features, which represent hidden relationships between users and movies.
	Predicted ratings for unseen movies are used to generate personalized recommendations.
Benefits
	Provides personalized movie suggestions for each user.
	Helps users discover new movies they might not have considered.
	Enhances user engagement and viewing satisfaction and efficiently handles large datasets of users and movies.
________________________________________

Singular Value Decomposition (SVD)
Singular Value Decomposition (SVD) is a matrix factorization technique that decomposes a matrix into three simpler matrices, revealing the latent structure in the data. It is widely used in recommender systems, dimensionality reduction, and noise reduction.
In the context of a movie recommendation system, SVD helps to factorize the user-item rating matrix into latent factors representing user preferences and item characteristics, which can then be used to predict missing ratings.
Mathematical Formulation:
For a given user-item rating matrix Rof size m×n(where mis the number of users and nis the number of movies):
R≈U" " Σ" " V^T
Where:
	Uis an m×kmatrix representing user latent features
	Σis a k×kdiagonal matrix containing singular values, which represent the strength of each latent feature
	V^Tis a k×nmatrix representing item latent features
	kis the number of latent factors (chosen based on the dataset)
Intuition:
	Each user and each item is represented as a vector in a k-dimensional latent space.
	The dot product of the user and item vectors approximates the rating a user would give to an item.
R ̂_ui=U_u⋅Σ⋅V_i^T
Where R ̂_uiis the predicted rating for user uand movie i.


________________________________________


Choosing the Algorithm for the Project
	The Singular Value Decomposition (SVD) algorithm was chosen because it is highly effective for collaborative filtering.
	Collaborative filtering relies on user-item interactions to make recommendations, and SVD is well-suited for this task.
	How SVD works:
	Factorizes the user-item matrix into three matrices (user features, singular values, item features).
	Captures latent features of users and items.
	Why SVD is ideal for a movie recommender system:
	Handles sparse data effectively.
	Predicts missing ratings for movies a user hasn’t rated.
	Improves accuracy by capturing latent factors like preferences for genres or themes.
	Scalable for large datasets.
	Enables personalized recommendations based on user behavior










________________________________________

Motivation and Reasons for Choosing SVD
	Handles missing data:
	User-item matrices are usually sparse; SVD can predict missing values efficiently.
	Dimensionality reduction:
	Reduces high-dimensional matrices into a smaller set of latent factors, preserving important patterns while improving efficiency.
	Preserves essential information:
	Captures the most significant features of users and items while ignoring noise.
	Computational efficiency:
	Reduces complexity and allows handling large datasets like 24M ratings.
	Proven success:
	Widely used in real-world recommendation systems like Netflix and Amazon.
	Captures latent relationships:
	Uncovers hidden correlations between users and movies, improving recommendations.









________________________________________
 Assumptions
	Computational feasibility:
	Full SVD computation is assumed feasible, but in practice, it may become computationally expensive for very large datasets.
	Meaningfulness of latent features:
	Assumes that latent features are meaningful, even though they may not have interpretable significance.
	Static user preferences:
	Assumes user preferences are constant over time, while in reality, tastes may evolve.
________________________________________
Model Evaluation and Techniques
	The Singular Value Decomposition (SVD) model was evaluated using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).
	4-fold cross-validation was used to assess the model’s performance.
	Results:
	RMSE values: approximately 0.988 to 1.004
	MAE values: approximately 0.786 to 0.805
	These metrics indicate that the SVD algorithm provides reasonably accurate predictions, as the error values are relatively low.
________________________________________
Inferences from Model Evaluation
	The SVD model predictions are highly accurate, as evidenced by the low RMSE and MAE values.
	Training time was reasonable, suggesting that the algorithm is suitable for large datasets like 24 million ratings.
	Some variability in testing time was observed, indicating that further optimization strategies (e.g., batch processing, dimensionality tuning) could improve efficiency.
________________________________________
Conclusion
The project demonstrates the development of a Movie Recommendation System using the Singular Value Decomposition (SVD) algorithm. By leveraging the user-item rating matrix and movie metadata, the system can predict missing ratings and provide personalized movie recommendations.
Key outcomes and insights include:
	Effective preprocessing and filtering allowed handling a large dataset of over 24 million ratings efficiently.
	The SVD algorithm successfully captured latent factors representing user preferences and movie characteristics, enabling accurate predictions.
	Model evaluation using RMSE and MAE showed low errors, indicating reliable performance for large-scale datasets.
	The system demonstrates scalability and adaptability, making it suitable for real-world applications like Netflix.
________________________________________

Future of Netflix Movie Recommendation System
The future of Netflix’s movie recommendation system lies in making recommendations more personalized, context-aware, and dynamic. Some key trends and advancements include:
	Enhanced Personalization:
	Using deep learning models to capture more complex user preferences, such as mood, viewing time, and interaction patterns.
	Integrating multi-modal data (movie trailers, posters, descriptions, subtitles) to understand user tastes beyond ratings.
	Context-Aware Recommendations:
	Recommendations may consider real-time context like the device being used, location, time of day, or current trends.
	Example: Suggesting family-friendly movies in the evening or short videos during commute times.
	Hybrid and Adaptive Systems:
	Combining collaborative filtering, content-based filtering, and reinforcement learning for smarter recommendations.
	Systems can adapt to changing user preferences over time, solving the static-preference limitation.
	Reducing Bias and Improving Diversity:
	Using algorithms to avoid repetitive suggestions and increase diversity, helping users explore new genres or international content.
	Preventing “filter bubbles” while still keeping recommendations relevant.
	Explainable Recommendations:
	Future systems may provide transparent reasoning for recommendations, e.g., “Recommended because you liked Movie X and Movie Y.”
	Helps increase user trust and engagement.
	Scalability and Real-Time Updates:
	With growing content libraries and users, Netflix will continue improving scalable algorithms and real-time updates to recommendations.
	Leveraging cloud computing and big data frameworks for instant adaptation to new ratings or releases.
In short: The future Netflix recommendation system will be more intelligent, adaptive, and user-centric, leveraging AI, deep learning, and real-time contextual data to deliver highly personalized and engaging movie suggestions.
________________________________________
Google Colab Link
https://colab.research.google.com/drive/1bgVrGUaPsAbgMmGfxV_QMw9bnZghjUd6#scrollTo=fqVaQ7biGg7e
________________________________________

