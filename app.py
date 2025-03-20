import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Load trained XGBoost model
model = load('final_xgb_music_recommendation_model.joblib')

# Load LabelEncoders to ensure proper encoding
user_encoder = load('user_encoder.joblib')
song_encoder = load('song_encoder.joblib')

# Load dataset to retrieve details (optional, improves recommendations)
data = pd.read_csv('/Users/ngoubimaximilliandiamgha/Desktop/kaggle_songs.txt',
                   sep='\t', header=None,
                   names=['song_id', 'song_name', 'artist_name', 'album'],
                   on_bad_lines='skip')

# Streamlit App Title and Description
st.title("🎵 Music Recommendation Dashboard")
st.markdown("""### Get personalized song recommendations based on user preferences and song popularity.""")

# User Inputs
user_id = st.text_input("Enter User ID (e.g., user_hash):")
song_id = st.text_input("Enter Song ID (e.g., song_hash):")
song_popularity = st.slider("Song Popularity", 0, 500, 50)
user_activity = st.slider("User Activity Level", 0, 500, 50)

# Button to predict
if st.button("Predict if User Likes Song"):
    if user_id and song_id:
        try:
            # Encode user and song IDs
            user_enc = user_encoder.transform([user_id])[0]
            song_enc = song_encoder.transform([song_id])[0]

            # Create input array
            input_data = np.array([[user_enc, song_enc, song_popularity, user_activity]])

            # Make prediction
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][prediction]

            # Display Results
            if prediction == 1:
                st.success(f"✅ User is likely to enjoy the song! (Confidence: {probability:.2%})")
            else:
                st.error(f"❌ User might not enjoy the song. (Confidence: {probability:.2%})")

            # Display song info if exists
            song_info = data[data['song_id'] == song_id]
            if not song_info.empty:
                st.write(f"**Song Name:** {song_info.iloc[0]['song_name']}")
                st.write(f"**Artist:** {song_info.iloc[0]['artist_name']}")
                st.write(f"**Album:** {song_info.iloc[0]['album']}")
            else:
                st.warning("Song details not found in database.")
        except Exception as e:
            st.error(f"⚠️ Error in prediction: {str(e)}")
    else:
        st.warning("⚠️ Please enter both User ID and Song ID.")
