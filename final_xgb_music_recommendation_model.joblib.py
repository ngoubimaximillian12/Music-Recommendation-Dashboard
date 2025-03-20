import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from joblib import dump

# Define file paths clearly
taste_profile_song_to_tracks_path = '/Users/ngoubimaximilliandiamgha/Desktop/taste_profile_song_to_tracks.txt'
unique_tracks_path = '/Users/ngoubimaximilliandiamgha/Desktop/unique_tracks.txt'
kaggle_triplets_path = '/Users/ngoubimaximilliandiamgha/Desktop/kaggle_visible_evaluation_triplets.txt'
kaggle_songs_path = '/Users/ngoubimaximilliandiamgha/Desktop/kaggle_songs.txt'

# Load datasets with careful delimiter handling
df_taste = pd.read_csv(
    taste_profile_song_to_tracks_path, sep='\t', header=None,
    names=['song_id', 'track_id'], on_bad_lines='skip')

df_tracks = pd.read_csv(
    unique_tracks_path, sep='<SEP>', header=None,
    names=['track_id', 'song_id_unique', 'artist_name', 'title'],
    engine='python', on_bad_lines='skip')

df_triplets = pd.read_csv(
    kaggle_triplets_path, sep='\t', header=None,
    names=['user_id', 'song_id', 'play_count'], on_bad_lines='skip')

df_songs = pd.read_csv(
    kaggle_songs_path, sep='\t', header=None,
    names=['song_id', 'song_name', 'artist_name', 'album'], on_bad_lines='skip')

# Merge carefully to avoid empty dataframe
merged_df = pd.merge(df_triplets, df_taste, on='song_id', how='inner')
merged_df = pd.merge(merged_df, df_tracks, on='track_id', how='inner')
merged_df = pd.merge(merged_df, df_songs, on='song_id', how='inner')

# Check merged_df content
print("Merged DF shape:", merged_df.shape)
print(merged_df.head())

# Drop only critical missing values clearly
merged_df.dropna(subset=['play_count', 'user_id', 'song_id'], inplace=True)

# Ensure data is present after cleaning
if merged_df.empty:
    raise ValueError("Merged dataframe is empty after cleaning. Check merges and data sources.")

# Feature engineering clearly
merged_df['song_popularity'] = merged_df.groupby('song_id')['play_count'].transform('sum')
merged_df['user_activity'] = merged_df.groupby('user_id')['play_count'].transform('sum')

# Label Encoding clearly
user_encoder = LabelEncoder()
song_encoder = LabelEncoder()

merged_df['user_encoded'] = user_encoder.fit_transform(merged_df['user_id'])
merged_df['song_encoded'] = song_encoder.fit_transform(merged_df['song_id'])

# Save encoders clearly
dump(user_encoder, 'user_encoder.joblib')
dump(song_encoder, 'song_encoder.joblib')

# Define features & binary target
X = merged_df[['user_encoded', 'song_encoded', 'song_popularity', 'user_activity']]
y = (merged_df['play_count'] > merged_df['play_count'].median()).astype(int)

# Verify X and y clearly before train-test split
print("X shape:", X.shape)
print("y distribution:", y.value_counts())

# Check again if X and y are valid
if X.empty or y.empty:
    raise ValueError("Feature matrix X or target y is empty. Cannot proceed.")

# Split data clearly and safely
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Train XGBoost model clearly
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Evaluate clearly
accuracy = xgb_model.score(X_test, y_test)
print(f"XGBoost model accuracy: {accuracy:.2%}")

# Save model clearly
dump(xgb_model, 'final_xgb_music_recommendation_model.joblib')
