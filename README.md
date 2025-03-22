# Music Enjoyment Prediction System

This project leverages historical listening data to predict whether a user will enjoy listening to specific songs. The system integrates various data science techniques such as **Exploratory Data Analysis (EDA)**, **Feature Engineering**, and **Machine Learning** models, including **Linear Regression**, **Ridge Classifier**, **Random Forest**, and **XGBoost** (selected model). The system also features a **real-time interactive dashboard** built with **Streamlit** for user interaction.

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Technologies Used](#technologies-used)
4. [Setup and Installation](#setup-and-installation)
5. [Usage](#usage)
6. [Model Training](#model-training)
7. [Real-Time Dashboard](#real-time-dashboard)
8. [Contributing](#contributing)
9. [License](#license)

---

## Overview

The Music Enjoyment Prediction System predicts whether a user will enjoy a song based on their historical listening behavior and metadata of the songs. It follows the end-to-end machine learning process:

- **Exploratory Data Analysis (EDA)**: Visualizing and understanding the dataset.
- **Feature Engineering**: Transforming raw data into features that improve model performance.
- **Machine Learning Algorithms**:
  - **Linear Regression** (baseline model)
  - **Ridge Classifier**
  - **Random Forest**
  - **XGBoost** (selected model)
- **Real-Time Interactive Dashboard**: Built with **Streamlit**, allowing users to interact with the model and visualize predictions in real-time.

---

## Dataset

The project uses the following datasets for training the model:

1. **User Listening Activity (`kaggle_visible_evaluation_triplets.txt`)**: Contains user-song interaction data (songs listened to by users).
2. **Unique Tracks (`unique_tracks.txt`)**: List of unique songs available in the music library.
3. **Songs Metadata (`kaggle_songs.txt`)**: Metadata about the songs, such as genre, artist, etc.
4. **Taste Profile (`taste_profile_song_to_tracks.txt`)**: Mapping of songs to user preferences, indicating enjoyment level.
5. **Users Information (`kaggle_users.txt`)**: Data about users including demographics and music preferences.

---

## Technologies Used

- **Python 3.x**: The main programming language for implementing machine learning models and the web application.
- **Streamlit**: A framework for building real-time interactive dashboards.
- **scikit-learn**: For machine learning algorithms like **Ridge Classifier**, **Random Forest**.
- **XGBoost**: A powerful machine learning library for regression and classification.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib/Seaborn**: For data visualization.
- **Jupyter Notebooks**: For data exploration and visualization.

---

## Setup and Installation

### 1. Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/music-enjoyment-prediction.git
cd music-enjoyment-prediction
