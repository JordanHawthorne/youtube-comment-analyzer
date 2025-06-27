#!/usr/bin/env python3
"""
Simple script to test YouTube API key functionality
"""

import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Load environment variables
load_dotenv()

def test_api_key(api_key=None):
    """Test if the YouTube API key is working"""
    
    if not api_key:
        api_key = os.getenv("YOUTUBE_API_KEY")
    
    if not api_key:
        print("❌ No API key found. Please set YOUTUBE_API_KEY in .env file or pass it as argument.")
        return False
    
    print(f"🔑 Testing API key: {api_key[:10]}...")
    
    try:
        # Build YouTube API client
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        # Test with a simple request - get a video's details
        test_video_id = "dQw4w9WgXcQ"  # Rick Astley - Never Gonna Give You Up
        
        request = youtube.videos().list(
            part="snippet",
            id=test_video_id
        )
        
        response = request.execute()
        
        if response['items']:
            video_title = response['items'][0]['snippet']['title']
            print(f"✅ API key is working! Test video: {video_title}")
            
            # Check quota
            request2 = youtube.videos().list(
                part="statistics",
                id=test_video_id
            )
            response2 = request2.execute()
            
            if response2['items']:
                view_count = response2['items'][0]['statistics']['viewCount']
                print(f"📊 Video has {int(view_count):,} views")
            
            return True
        else:
            print("❌ API key seems valid but couldn't fetch video data")
            return False
            
    except HttpError as e:
        if e.resp.status == 403:
            print(f"❌ API Key Error (403): {e.reason}")
            print("   Possible causes:")
            print("   - Invalid API key")
            print("   - API key doesn't have YouTube Data API v3 enabled")
            print("   - Quota exceeded")
            print("   - IP/referrer restrictions")
        elif e.resp.status == 400:
            print(f"❌ Bad Request (400): {e.reason}")
        else:
            print(f"❌ HTTP Error {e.resp.status}: {e.reason}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    import sys
    
    # Allow passing API key as command line argument
    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    
    if api_key:
        print("Using API key from command line argument")
    else:
        print("Using API key from .env file")
    
    test_api_key(api_key) 