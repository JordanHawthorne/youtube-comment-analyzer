import streamlit as st
import os
import sqlite3
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from sentence_transformers import SentenceTransformer
import hdbscan
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yake
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# --- Session State for API Key Management ---
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.getenv("YOUTUBE_API_KEY", "")

# --- Database Functions ---
def init_db():
    """Initializes the SQLite database."""
    conn = sqlite3.connect('youtube_comments.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY,
            video_title TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS comments (
            comment_id TEXT PRIMARY KEY,
            video_id TEXT,
            text TEXT,
            author TEXT,
            timestamp TEXT,
            FOREIGN KEY (video_id) REFERENCES videos (video_id)
        )
    ''')
    conn.commit()
    conn.close()

@st.cache_data
def cache_comments(video_id, comments):
    """Caches comments into the SQLite database."""
    conn = sqlite3.connect('youtube_comments.db')
    c = conn.cursor()
    # Get video title (assuming first comment fetch also gets title)
    # In a real app, you might fetch video details separately
    video_title = f"Title for {video_id}" 
    try:
        c.execute("INSERT OR IGNORE INTO videos (video_id, video_title) VALUES (?, ?)", (video_id, video_title))
        for comment in comments:
            c.execute("INSERT OR IGNORE INTO comments (comment_id, video_id, text, author, timestamp) VALUES (?, ?, ?, ?, ?)",
                      (comment['id'], video_id, comment['text'], comment['author'], comment['timestamp']))
        conn.commit()
    finally:
        conn.close()

@st.cache_data
def get_cached_comments(video_id):
    """Retrieves cached comments from the SQLite database."""
    conn = sqlite3.connect('youtube_comments.db')
    try:
        df = pd.read_sql_query("SELECT text, author, timestamp FROM comments WHERE video_id = ?", conn, params=(video_id,))
        return df if not df.empty else None
    finally:
        conn.close()

# --- YouTube API Functions ---
@st.cache_data
def fetch_comments(video_id: str, api_key: str):
    """Fetches all public comments and replies for a given YouTube video ID."""
    if not api_key:
        st.error("YouTube API key not found. Please enter it in the sidebar.")
        return []

    youtube = build('youtube', 'v3', developerKey=api_key)
    comments_list = []

    try:
        # Fetch top-level comments
        request = youtube.commentThreads().list(
            part='snippet,replies',
            videoId=video_id,
            maxResults=100,
            textFormat='plainText'
        )
        while request:
            response = request.execute()
            for item in response['items']:
                snippet = item['snippet']['topLevelComment']['snippet']
                comments_list.append({
                    'id': item['snippet']['topLevelComment']['id'],
                    'text': snippet['textDisplay'],
                    'author': snippet['authorDisplayName'],
                    'timestamp': snippet['publishedAt']
                })
                if 'replies' in item:
                    for reply_item in item['replies']['comments']:
                        reply_snippet = reply_item['snippet']
                        comments_list.append({
                            'id': reply_item['id'],
                            'text': reply_snippet['textDisplay'],
                            'author': reply_snippet['authorDisplayName'],
                            'timestamp': reply_snippet['publishedAt']
                        })
            request = youtube.commentThreads().list_next(request, response)
    except HttpError as e:
        if e.resp.status == 403:
            st.error(f"API Key Error: {e.reason}. This usually means:\n"
                     f"- Your API key is invalid\n"
                     f"- You've exceeded your quota\n"
                     f"- The API key doesn't have YouTube Data API v3 enabled")
        elif e.resp.status == 404:
            st.error(f"Video not found. Please check the video ID: {video_id}")
        else:
            st.error(f"YouTube API Error: {e}")
        return []
    except Exception as e:
        st.error(f"Unexpected error fetching comments: {str(e)}")
        return []

    return comments_list

# --- Analysis Functions ---
@st.cache_resource
def get_embedding_model():
    """Loads the sentence-transformer model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def analyze_topics(_comments_df):
    """Analyzes comments to find topics, sentiment, and keywords."""
    if _comments_df.empty or _comments_df['text'].nunique() < 2:
        return pd.DataFrame(), {}, {}, pd.DataFrame()

    # 1. Clean and Deduplicate
    cleaned_texts = _comments_df['text'].drop_duplicates().tolist()

    # 2. Generate Embeddings
    model = get_embedding_model()
    with st.spinner("Generating comment embeddings..."):
        embeddings = model.encode(cleaned_texts, show_progress_bar=True)

    # 3. Cluster Embeddings with HDBSCAN
    with st.spinner("Clustering comments to find topics..."):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2, gen_min_span_tree=True)
        cluster_labels = clusterer.fit_predict(embeddings)

    # Combine results into a DataFrame
    clustered_df = pd.DataFrame({'text': cleaned_texts, 'topic': cluster_labels})
    
    # 4. Sentiment Analysis with VADER
    analyzer = SentimentIntensityAnalyzer()
    sentiments = _comments_df['text'].apply(lambda text: analyzer.polarity_scores(text))
    sentiment_df = pd.DataFrame(list(sentiments))
    
    def classify_sentiment(row):
        if row['compound'] >= 0.05:
            return 'Positive'
        elif row['compound'] <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    
    sentiment_counts = sentiment_df.apply(classify_sentiment, axis=1).value_counts().to_dict()

    # 5. Keyword Extraction with YAKE
    full_text = " ".join(_comments_df['text'].tolist())
    kw_extractor = yake.KeywordExtractor(n=1, dedupLim=0.9, top=20, features=None)
    keywords = kw_extractor.extract_keywords(full_text)
    keywords_dict = {kw: score for kw, score in keywords}

    return clustered_df, sentiment_counts, keywords_dict, _comments_df

# --- Dashboard and UI Functions ---
def build_dashboard(clustered_df, sentiment_counts, keywords, all_comments_df):
    """Builds the Streamlit dashboard to display analysis results."""
    st.header("📊 Analysis Dashboard")

    if clustered_df.empty:
        st.warning("Not enough unique comments to generate a dashboard.")
        return

    # --- Topic Analysis Section ---
    st.subheader("💬 Top Discussion Topics")
    topic_counts = clustered_df['topic'].value_counts()
    # Exclude noise (-1)
    top_topics = topic_counts[topic_counts.index != -1].head(10)

    if not top_topics.empty:
        # Create representative names for topics
        topic_names = {}
        for topic_id in top_topics.index:
            topic_texts = clustered_df[clustered_df['topic'] == topic_id]['text'].tolist()
            # Simple name: first 5 words of the first comment
            topic_names[topic_id] = ' '.join(topic_texts[0].split()[:5]) + '...'

        fig, ax = plt.subplots()
        ax.barh([topic_names[i] for i in top_topics.index], top_topics.values, color='skyblue')
        ax.set_xlabel("Number of Comments")
        ax.set_title("Top Comment Clusters")
        st.pyplot(fig)
    else:
        st.info("No significant topics were identified from the comments.")

    # --- Sentiment & Keywords Section ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("😊 Sentiment Distribution")
        if sentiment_counts:
            fig, ax = plt.subplots()
            ax.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%',
                   colors=['lightgreen', 'lightcoral', 'lightgrey'])
            ax.axis('equal')
            st.pyplot(fig)
        else:
            st.info("Could not determine sentiment.")

    with col2:
        st.subheader("🔑 Frequent Keywords")
        if keywords:
            st.table(pd.DataFrame(list(keywords.items()), columns=['Keyword', 'Score']))
        else:
            st.info("No significant keywords found.")

    # --- FAQ Generation ---
    st.subheader("❓ Auto-Generated FAQ")
    if not top_topics.empty:
        for topic_id in top_topics.index:
            with st.expander(f"**Theme: {topic_names[topic_id]}** ({top_topics[topic_id]} comments)"):
                topic_comments = clustered_df[clustered_df['topic'] == topic_id]['text'].head(3)
                for comment in topic_comments:
                    st.markdown(f"- *{comment}*")
    else:
        st.warning("No topics found to generate an FAQ.")

    # --- Script Generation ---
    st.subheader("🎬 Generate YouTube Shorts Script")
    if st.button("Generate 60-Second Script"):
        generate_script(top_topics, clustered_df, topic_names)

def generate_script(top_topics, clustered_df, topic_names):
    """Generates and displays a 60-second video script."""
    st.markdown("### 📜 Your 60-Second FAQ Video Script")
    
    if top_topics.empty or len(top_topics) < 3:
        st.warning("Not enough topics to create a full script. Try a video with more comments!")
        return

    # --- Script Content ---
    script = f"""
**Hook (0-5 seconds):**
(Upbeat, engaging music starts)
**Host:** "Ever wondered what everyone's REALLY talking about in the comments? I analyzed thousands of comments on my last video, and you won't BELIEVE what I found! Here are the top 3 things on your mind."

**Point 1 (5-20 seconds):**
**Host:** "First up, a lot of you were asking about **{topic_names[top_topics.index[0]]}**. The main question seems to be..."
*(Show a representative comment on screen)*
> '{clustered_df[clustered_df['topic'] == top_topics.index[0]]['text'].iloc[0]}'
**Host:** "And here's the simple answer: [Your concise answer to the first theme here]."

**Point 2 (20-35 seconds):**
**Host:** "Next, there was a huge discussion around **{topic_names[top_topics.index[1]]}**. It's clear that many of you feel..."
*(Show another key comment)*
> '{clustered_df[clustered_df['topic'] == top_topics.index[1]]['text'].iloc[0]}'
**Host:** "My take on this is: [Your clear, helpful response to the second theme]."

**Point 3 (35-50 seconds):**
**Host:** "And finally, let's talk about **{topic_names[top_topics.index[2]]}**. This one was surprising!"
*(Display a third insightful comment)*
> '{clustered_df[clustered_df['topic'] == top_topics.index[2]]['text'].iloc[0]}'
**Host:** "To clear things up: [Your definitive answer to the third theme]."

**Call to Action (50-60 seconds):**
**Host:** "Did I miss anything? Drop a comment below with what you want me to analyze next! And don't forget to like and subscribe for more insights!"
(Upbeat music swells, end screen with channel name and subscribe button)
"""
    st.code(script, language='markdown')


# --- Main App Logic ---
def main():
    """Main function to run the Streamlit app."""
    st.title("🚀 YouTube Comment Analyzer & Script Generator")
    
    init_db() # Ensure DB is ready

    # Sidebar for API Key Management
    with st.sidebar:
        st.header("⚙️ Settings")
        api_key_input = st.text_input(
            "YouTube API Key:",
            value=st.session_state.api_key,
            type="password",
            help="Enter your YouTube Data API v3 key"
        )
        
        if api_key_input != st.session_state.api_key:
            st.session_state.api_key = api_key_input
            st.success("API Key updated!")
        
        if not st.session_state.api_key:
            st.warning("⚠️ Please enter your YouTube API key to use this app.")
            st.markdown("""
            **How to get an API key:**
            1. Go to [Google Cloud Console](https://console.cloud.google.com/)
            2. Create a new project or select existing
            3. Enable YouTube Data API v3
            4. Create credentials (API Key)
            5. Copy and paste the key above
            """)

    st.markdown("""
    Welcome! This tool analyzes YouTube comments to uncover key topics, sentiment, and auto-generates a video script.
    
    **To get started:**
    1.  Enter a YouTube video URL below.
    2.  Click "Analyze Comments".
    3.  Explore the dashboard!
    
    **Note:** The first analysis for a video may take a few minutes. Results are cached for faster re-analysis.
    """)

    youtube_url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")

    if st.button("Analyze Comments"):
        if not youtube_url:
            st.warning("Please enter a YouTube URL.")
            return
        
        if not st.session_state.api_key:
            st.error("Please enter your YouTube API key in the sidebar first!")
            return
        
        try:
            # Extract video ID
            video_id = youtube_url.split("v=")[1].split("&")[0]
        except IndexError:
            st.error("Invalid YouTube URL. Please use a format like '...watch?v=VIDEO_ID'")
            return

        with st.spinner(f"Checking cache for video ID: {video_id}..."):
            comments_df = get_cached_comments(video_id)

        if comments_df is None:
            with st.spinner("No cache found. Fetching comments from YouTube API..."):
                comments_list = fetch_comments(video_id, st.session_state.api_key)
                if not comments_list:
                    st.error("Could not fetch comments. Check the video URL and your API key.")
                    return
                
                comments_df = pd.DataFrame(comments_list)
                cache_comments(video_id, comments_list)
                st.success(f"Fetched and cached {len(comments_list)} comments.")
        else:
            st.success(f"Loaded {len(comments_df)} comments from cache.")

        # Perform analysis
        clustered_df, sentiment_counts, keywords, all_comments_df = analyze_topics(comments_df)
        
        # Store analysis results in session state to avoid re-computing on every interaction
        st.session_state['analysis_complete'] = True
        st.session_state['clustered_df'] = clustered_df
        st.session_state['sentiment_counts'] = sentiment_counts
        st.session_state['keywords'] = keywords
        st.session_state['all_comments_df'] = all_comments_df

    # Display dashboard if analysis has been run
    if 'analysis_complete' in st.session_state and st.session_state['analysis_complete']:
        build_dashboard(
            st.session_state['clustered_df'],
            st.session_state['sentiment_counts'],
            st.session_state['keywords'],
            st.session_state['all_comments_df']
        )

if __name__ == "__main__":
    main()
