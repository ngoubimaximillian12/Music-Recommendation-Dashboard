import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(
    page_title="Music Recommendation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Data
@st.cache_data
def load_data():
    songs_df = pd.read_csv(
        '/Users/ngoubimaximilliandiamgha/Desktop/kaggle_songs.txt',
        sep='\t', header=None,
        names=['song_id', 'song_name', 'artist_name', 'album'],
        on_bad_lines='skip'
    )

    triplets_df = pd.read_csv(
        '/Users/ngoubimaximilliandiamgha/Desktop/kaggle_visible_evaluation_triplets.txt',
        sep='\t', header=None,
        names=['user_id', 'song_id', 'play_count'],
        on_bad_lines='skip'
    )
    merged_df = triplets_df.merge(songs_df, on='song_id', how='left')
    merged_df.dropna(inplace=True)
    return merged_df

df = load_data()

# Dashboard Title
st.title("🎶 Music Recommendation Dashboard")

# Sidebar Filters
st.sidebar.header("🔍 Filters")
top_n_songs = st.sidebar.slider("Top N Songs", min_value=5, max_value=30, value=10)
top_n_artists = st.sidebar.slider("Top N Artists", min_value=5, max_value=30, value=10)

# Main Dashboard Components
col1, col2 = st.columns(2)

# Top N Songs by Play Count
with col1:
    st.subheader(f"🎧 Top {top_n_songs} Songs by Play Count")
    top_songs = df.groupby('song_name')['play_count'].sum().sort_values(ascending=False).head(top_n_songs)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(y=top_songs.index, x=top_songs.values, palette='viridis', ax=ax)
    ax.set_xlabel('Play Count')
    ax.set_ylabel('Song Name')
    st.pyplot(fig)

# Top N Artists by Play Count
with col2:
    st.subheader(f"🎤 Top {top_n_artists} Artists by Play Count")
    top_artists = df.groupby('artist_name')['play_count'].sum().sort_values(ascending=False).head(top_n_artists)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(y=top_artists.index, x=top_artists.values, palette='rocket', ax=ax2)
    ax2.set_xlabel('Play Count')
    ax2.set_ylabel('Artist Name')
    st.pyplot(fig2)

# User-Song Interaction Analysis
st.subheader("👤 User-Song Interaction Overview")
selected_user = st.text_input("Enter a User ID to see their listening history:")

if selected_user:
    user_history = df[df['user_id'] == selected_user]

    if not user_history.empty:
        st.write(f"### Listening History for User ID: {selected_user}")
        st.dataframe(user_history[['song_name', 'artist_name', 'play_count']].sort_values(by='play_count', ascending=False).head(20))

        fig3, ax3 = plt.subplots(figsize=(12, 5))
        sns.histplot(user_history['play_count'], bins=30, kde=True, color='skyblue', ax=ax3)
        ax3.set_title(f"Play Count Distribution for User: {selected_user}")
        ax3.set_xlabel("Play Count")
        st.pyplot(fig3)
    else:
        st.warning("No listening history found for this user ID.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit by Maximillian Diangha")

