Music Enjoyment Prediction System
This project leverages historical listening data to predict whether a user will enjoy listening to specific songs. The system integrates various data science techniques such as Exploratory Data Analysis (EDA), Feature Engineering, and Machine Learning models, including Linear Regression, Ridge Classifier, Random Forest, and XGBoost (selected model). The system also features a real-time interactive dashboard built with Streamlit for user interaction.

Table of Contents
Overview

Dataset

Technologies Used

Setup and Installation

Usage

Model Training

Real-Time Dashboard

Contributing

License

Overview
This system is designed to predict whether a user will enjoy a specific song based on their listening history and song metadata. The project follows a complete data science workflow:

Exploratory Data Analysis (EDA): Understanding the dataset and visualizing patterns and distributions.

Feature Engineering: Creating new features from raw data to improve model performance.

Machine Learning Algorithms:

Linear Regression: Used as a baseline model for predictions.

Ridge Classifier: To apply regularization for better generalization.

Random Forest: An ensemble method that improves prediction accuracy.

XGBoost: Selected as the final model due to its high performance in classification tasks.

Real-Time Interactive Dashboard: Built with Streamlit for an interactive user interface.

Dataset
The dataset used in this project is sourced from Kaggle and includes the following files:

User Listening Activity (kaggle_visible_evaluation_triplets.txt): Contains historical data of user-song interactions, including which songs were listened to by which users.

Unique Tracks (unique_tracks.txt): A list of unique songs available in the music library.

Songs Metadata (kaggle_songs.txt): Metadata about the songs, such as genre, artist, and other attributes.

Taste Profile (taste_profile_song_to_tracks.txt): A mapping between songs and user preferences, indicating whether a user enjoyed the song.

Users Information (kaggle_users.txt): Data about users, including demographics and preferences.

Technologies Used
This project makes use of several key technologies:

Python 3.x: The primary programming language for implementing the machine learning models and dashboard.

Pandas: Data manipulation and preprocessing library.

NumPy: Library for numerical computations and handling arrays.

Matplotlib/Seaborn: Libraries for data visualization.

scikit-learn: For machine learning algorithms and model evaluation.

XGBoost: An optimized gradient boosting library for classification tasks.

Streamlit: A framework for building real-time interactive web applications for data science projects.

Jupyter Notebooks: For prototyping and data analysis.

