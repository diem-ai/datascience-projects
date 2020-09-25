Hydrid Movie Recommendation system:
  - It is combination of Item-item collaborative filtering and content based model
  - Item item similarity: It uses the most similar items to a user's already-rated items to generate a list of recommendations. The result is the cosine of those rating vectors.
  - Content based similarity: This method bases on the information of movies such as: genres, stars and crew.
  - Both models use truncated singular value decomposition (SVD) to overcome dimensionality curse
  
Project structure:
   |___data
          |___ normalised_movies.csv
          |___ ratings_matrix.csv
          |___ content_matrix.csv
          |___ metadata_movies.csv
          |___ ratings.csv
          |___ credits.csv
          |___ keywords.csv
   |___ HybridRecommendationModel.ipynb
   |___ HydridModelEvaluation.ipynb
   |___ HybridRecommendationModel.html
   |___ HydridModelEvaluation.html
   |___ recommender.py
  
 Data/Files Notes:
  - metadata_movies.csv, ratings.csv, credits.csv and keywords can be found on Kaggle.com
  - normalised_movies.csv, ratings_matrix.csv and content_matrix.csv are generated when training model: HybridRecommendationModel.ipynb
  - Those files are the input for HydridModelEvaluation.ipynb
  - recommender.py consist of 2 python classes: 
      - class Utility: help for cleaning data, showing top movies with right format, looking for id/index of movie
      - class Similarity: compute the cosine similariy between sparse matrix (ratings_matrix.csv or content_matrix.csv) and given movie
  - Because of large datasets, all note books are run on Colab. If you want to run them on local, check out the below "Run on Local"
  - HybridRecommendationModel.html and HydridModelEvaluation.html: for reading notebooks on web browser.
  
 Requirements:
  - Python >= 3.7
  - Jupyter Notebook

Dependencies:
  - pandas
  - sklearn
  - numpy
  - ast
  - matplotlib
  - nltk

Installation
 - nltk : pip install nltk
 - Sklearn: pip install sklearn
 - pandas: pip install pandas
 - numpy: pip install numpy
 - matplotlib: pip install matplotlib

Run the projecy on local:
 - Clone the project: by downloading or git command: git clone https://github.com/diem-ai/movie-recommender.git
 - Install libraries in Requirements and Dependencies
 - Dowload datasets (on Kaggle) and put them on /data directory
 - Comment the colab setup part (first part) and change the file path
 
Improvements (next commits):
 - Build a simple webapp for simulation
 - Applying deeplearning
 - Evaluate models with Suprise 

Acknowledgements:
 Got Datasets from Kaggle.com
