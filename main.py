from pytubefix import YouTube
from pytubefix.exceptions import VideoUnavailable
import whisper
from textblob import TextBlob
from moviepy import * 
import numpy as np
import spacy
import nltk
from nltk.corpus import opinion_lexicon
import requests
import csv
from io import StringIO

def get_emotional_keywords(threshold=0.5):
    """Get words associated with excitement, surprise and other highlight-related emotions."""
    # Download NLTK resources if needed
    nltk.download('opinion_lexicon', quiet=True)
    
    # First, get positive words from NLTK
    positive_words = set(opinion_lexicon.positive())
    
    # Download the NRC emotion lexicon
    url = "https://saifmohammad.com/WebDocs/NRC-Emotion-Lexicon.zip"
    try:
        # This might not work if the website changes or blocks automated downloads
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception("Failed to download NRC lexicon")
    except:
        print("Could not download NRC lexicon automatically. Please download manually.")
        return list(positive_words)[:100]  # Fallback to just positive words
        
    # Process the lexicon
    emotion_words = {
        'joy': [],
        'surprise': [],
        'anticipation': [],
        'trust': []
    }
    
    # Parse the CSV
    csv_reader = csv.reader(StringIO(response.text), delimiter='\t')
    for row in csv_reader:
        if len(row) < 2:
            continue
            
        word = row[0]
        emotion = row[1]
        score = float(row[2]) if len(row) > 2 else 0
        
        if emotion in emotion_words and score >= threshold:
            emotion_words[emotion].append(word)
    
    # Combine words from different emotions, prioritizing words that appear in multiple categories
    all_emotion_words = {}
    for emotion, words in emotion_words.items():
        for word in words:
            all_emotion_words[word] = all_emotion_words.get(word, 0) + 1
    
    # Sort by frequency across emotion categories
    sorted_words = sorted(all_emotion_words.items(), key=lambda x: x[1], reverse=True)
    highlight_keywords = [word for word, count in sorted_words]
    
    # Include some positive words
    highlight_keywords.extend([word for word in positive_words if word not in highlight_keywords][:50])
    
    return highlight_keywords[:200]  

def download_video(url):
    try:
        yt = YouTube(
            url,
            use_oauth=False,
            allow_oauth_cache=True
        )
        print(f"Title: {yt.title}")
        # New way to get the highest resolution stream
        stream = yt.streams.filter(progressive=True, file_extension='mp4').get_highest_resolution()
        stream.download(filename="input_video.mp4")
        print("Download successful!")
    except VideoUnavailable:
        print(f"Video {url} is unavailable")
    except Exception as e:
        print(f"Error: {str(e)}")

def get_transcription():
    model = whisper.load_model("base")  # Use 'small' or 'medium' for better accuracy
    result = model.transcribe("input_video.mp4")
    return result["segments"]

def find_highlight_segments(segments, top_n=3, min_duration=3, max_duration=30):
    """
    Find the most engaging segments based on multiple factors:
    - Sentiment intensity (both positive and negative can be engaging)
    - Presence of key phrases indicating important moments
    - Speaking rate/energy
    - Segment length (not too short, not too long)
    """
    # Load spaCy for better text analysis
    nlp = spacy.load("en_core_web_sm")
    
    # Keywords that might indicate interesting content
    highlight_keywords = get_emotional_keywords()
    
    scored_segments = []
    
    for segment in segments:
        # Basic info
        text = segment["text"]
        start_time = segment["start"]
        end_time = segment["end"]
        duration = end_time - start_time
        
        # Skip segments that are too short or too long
        if duration < min_duration or duration > max_duration:
            continue
            
        # Sentiment analysis using TextBlob
        analysis = TextBlob(text)
        
        # We care about sentiment intensity (both positive and negative can be interesting)
        sentiment_score = abs(analysis.sentiment.polarity) * 2
        
        # More detailed text analysis with spaCy
        doc = nlp(text)
        
        # Check for presence of highlight keywords
        keyword_score = sum(1 for word in doc if word.text.lower() in highlight_keywords) * 0.5
        
        # Check for named entities (people, organizations, etc.)
        entity_score = len(doc.ents) * 0.3
        
        # Consider speaking rate/energy (words per second)
        word_count = len(text.split())
        speaking_rate = word_count / max(duration, 1)  # Avoid division by zero
        speaking_rate_score = min(speaking_rate / 3, 1) * 0.7  # Normalize and cap
        
        # Calculate final score with different weights
        score = (
            sentiment_score * 1.0 +
            keyword_score * 1.5 +
            entity_score * 1.0 +
            speaking_rate_score * 0.8 +
            (word_count / 20) * 0.5  # Some bias toward longer segments, but capped
        )
        
        scored_segments.append((segment["start"], score))
    
    # Sort by score and return top N segments
    top_segments = sorted(scored_segments, key=lambda x: x[1], reverse=True)[:top_n]
    return top_segments

def create_short(highlights, duration=60):
    video = VideoFileClip("input_video.mp4")
    clips = []
    
    # Create clips from each highlight
    for start_time, _ in highlights:
        clip = video.subclipped(start_time, min(start_time + duration/len(highlights), video.end))
        clips.append(clip)
    
    # Concatenate all clips
    final_clip = clips[0]
    for clip in clips[1:]:
        final_clip = concatenate_videoclips([final_clip, clip])
    
    # Convert to vertical (9:16)
    final_clip = final_clip.resized(height=1920)
    final_clip = final_clip.cropped(x_center=final_clip.w/2, y_center=final_clip.h/2, width=1080, height=1920)

    final_clip = CompositeVideoClip([final_clip])
    
    final_clip.write_videofile("short_highlights.mp4", fps=24)

if __name__ == "__main__":
    video_url = 'https://www.youtube.com/watch?v=eUDGlxu_-ic'
    download_video(video_url)
    print("Video downloaded successfully.")
    segments = get_transcription()

    # Find highlights
    highlights = find_highlight_segments(segments)

    # Create single short with all highlights
    create_short(highlights)
        