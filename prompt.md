@system: You are a Google Gemini 2.5 Pro instance with advanced code generation and data integration capabilities. Your task is to build a local app from scratch using Streamlit, designed for YouTube comment analysis and automatic FAQ/script creation.

@task:
Build a fully local Streamlit app that:
1. Accepts a YouTube video link from the user.
2. Uses the YouTube Data API v3 to fetch all public comments and replies.
3. Caches results in a local SQLite database for reusability.
4. Performs intelligent comment analysis:
   - Cleans and deduplicates text
   - Generates embeddings using `all-MiniLM-L6-v2` via sentence-transformers
   - Clusters embeddings using HDBSCAN to find main discussion themes
   - Extracts sentiment using VADER
   - Surfaces frequent keywords using YAKE
5. Displays an interactive dashboard with:
   - Bar chart of top topics
   - Sentiment pie chart
   - Auto-generated expandable FAQ section
   - Button to generate and display a 60-second “YouTube Shorts-style” script based on top user concerns
6. Exports:
   - `app.py` file (Streamlit app)
   - `requirements.txt`
   - `README.md` with local run instructions

@inputs:
@youtube {Insert_YouTube_Link_Here}

@instructions:
- Use Python 3.11, Streamlit ≥1.33, and the following libraries: `google-api-python-client`, `sentence-transformers`, `hdbscan`, `pandas`, `matplotlib`, `vaderSentiment`, `yake`, `sqlite3`.
- Organize code with these key functions: `fetch_comments()`, `cache_comments()`, `analyze_topics()`, `build_dashboard()`.
- Use `st.cache_data` or `st.cache_resource` decorators to optimize performance.
- Present your code in fully usable Python blocks and wrap explanations in Markdown-friendly headers like `### How It Works`.
- Conclude with:
   - @output: a block showing a fully narrated 60-second FAQ video script, starting with a hook, addressing 3 top comment themes, and ending with a CTA.
   - @next_steps: Suggest future extensions like Docker deployment or cloud hosting.

@notes:
Make sure to isolate the YouTube API key in a `.env` file and mention `.gitignore` best practices. Prioritize readability, UX, and component separation.

@format: Markdown + Code Blocks + Narrative + JSON Outputs if needed.