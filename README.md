
# YouTube Comment Analyzer & Script Generator

This Streamlit application analyzes comments from a YouTube video to identify key discussion topics, sentiment, and frequent keywords. It then uses this analysis to generate a 60-second "YouTube Shorts-style" video script to address the main themes found in the comments.

## Features

- **YouTube Comment Fetching**: Retrieves all public comments and replies from a given video URL using the YouTube Data API v3.
- **Local Caching**: Caches comments in a local SQLite database to speed up subsequent analyses of the same video.
- **Topic Clustering**: Uses `sentence-transformers` and `HDBSCAN` to cluster comments and identify major discussion themes.
- **Sentiment Analysis**: Analyzes the sentiment of comments using `VADER`.
- **Keyword Extraction**: Extracts the most frequent keywords using `YAKE`.
- **Interactive Dashboard**: Displays the analysis results in an easy-to-understand Streamlit dashboard, including charts for topics and sentiment.
- **Auto-Generated FAQ**: Creates an expandable FAQ section based on the identified comment clusters.
- **AI Script Generation**: Generates a 60-second video script based on the top three comment themes, complete with a hook, main points, and a call to action.

## How It Works

1.  **Input**: The user provides a YouTube video URL.
2.  **Fetch & Cache**: The app extracts the video ID, fetches all comments via the YouTube API, and caches them in a local `youtube_comments.db` file. If the video has been analyzed before, it loads the comments from the cache.
3.  **Analysis**:
    *   Comment text is cleaned and deduplicated.
    *   The `all-MiniLM-L6-v2` model generates embeddings for each unique comment.
    *   `HDBSCAN` clusters these embeddings to group similar comments into topics.
    *   `VADER` calculates the sentiment (Positive, Negative, Neutral) for each comment.
    *   `YAKE` extracts the most relevant keywords from the entire comment corpus.
4.  **Display**: The results are presented in an interactive dashboard.
5.  **Script Generation**: Upon request, a formatted 60-second video script is generated to address the top user concerns and questions.

## Setup and Local Run Instructions

### Prerequisites

- Python 3.11
- A YouTube Data API v3 Key

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/JordanHawthorne/youtube-comment-analyzer.git
    cd youtube-comment-analyzer
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    *   Create a `.env` file by copying the example template:
        ```bash
        cp .env.example .env
        ```
    *   Open the new `.env` file and replace the placeholder `YOUR_YOUTUBE_API_KEY_HERE` with your actual YouTube Data API v3 key.

### Running the Application

1.  **Run the Streamlit app from your terminal:**
    ```bash
    streamlit run app.py
    ```

2.  **Open your web browser** to the local URL provided by Streamlit (usually `http://localhost:8501`).

3.  **Paste a YouTube video URL** into the input box and click "Analyze Comments" to start.

## Troubleshooting

### Common Issues

1. **"API key invalid" error on new video links:**
   - Check if your API key has daily quota limits (default is 10,000 units/day)
   - Verify the API key is still active in Google Cloud Console
   - Make sure YouTube Data API v3 is enabled for your project
   - Check if the API key has IP restrictions that might be blocking requests

2. **"Video not found" error:**
   - Ensure the video is public and not age-restricted
   - Check if the video ID is correctly extracted from the URL
   - Try using the standard YouTube URL format: `https://www.youtube.com/watch?v=VIDEO_ID`

3. **No comments fetched:**
   - The video might have comments disabled
   - The video might be too new and comments haven't loaded yet
   - Check your internet connection

4. **Session state issues:**
   - If the API key keeps resetting, try refreshing the page
   - Clear your browser cache if issues persist

## Next Steps & Future Extensions

- **Docker Deployment**: Package the application into a Docker container for easier deployment and environment management.
- **Cloud Hosting**: Deploy the application to a cloud service like Streamlit Community Cloud, Heroku, or AWS so it can be accessed by a wider audience.
- **Advanced Topic Naming**: Use a large language model (LLM) to generate more descriptive and accurate names for the identified comment clusters.
- **Time-Series Analysis**: Plot comment activity over time to identify spikes in engagement.
- **User-Specific Analysis**: Analyze comments from a specific user across multiple videos.
