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
from nrclex import NRCLex
from moviepy.video.fx import FadeIn, FadeOut

def get_emotional_keywords_v2(target_emotions=None):
    """
    Get emotional keywords from NRCLex lexicon.
    
    Args:
        target_emotions (list): List of emotions to target (e.g., ['joy', 'surprise'])
        
    Returns:
        list: Words associated with the target emotions from NRCLex lexicon
    """
    if target_emotions is None:
        target_emotions = ['joy', 'surprise', 'anticipation', 'trust']
    
    # Create an empty NRCLex object to access the lexicon
    emotion_analyzer = NRCLex('')
    
    # Get all words from the lexicon
    emotional_words = []
    
    # Iterate through the affect dictionary to find words with target emotions
    for word, emotions in emotion_analyzer.affect_dict.items():
        # Check if the word has any of our target emotions
        if any(emotion in target_emotions for emotion in emotions):
            emotional_words.append(word)
    
    return emotional_words

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
    - Time position (early and late segments get a boost)
    """
    # Load spaCy for better text analysis
    nlp = spacy.load("en_core_web_sm")
    
    # Get video duration for time-based scoring
    video = VideoFileClip("input_video.mp4")
    video_duration = video.duration
    video.close()
    
    # Keywords that might indicate interesting content
    highlight_keywords = get_emotional_keywords_v2()
    
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
        
        # Calculate base score with different weights
        score = (
            sentiment_score * 1.0 +
            keyword_score * 1.5 +
            entity_score * 1.0 +
            speaking_rate_score * 0.8 +
            (word_count / 20) * 0.5  # Some bias toward longer segments, but capped
        )
        
        # Add time-based scoring (peaks in first 30s and last 30s score higher)
        time_position = start_time / video_duration
        if time_position < 0.3 or time_position > 0.7:
            score *= 1.3  # Boost early and late segments
        
        scored_segments.append((segment["start"], score))
    
    # Sort by score and return top N segments
    top_segments = sorted(scored_segments, key=lambda x: x[1], reverse=True)[:top_n]
    return top_segments

def create_short(highlights, duration=60):
    video = VideoFileClip("input_video.mp4")
    
    # Sort highlights by time and ensure non-overlapping
    highlights.sort(key=lambda x: x[0])
    final_segments = []
    current_end = 0
    
    for start, score in highlights:
        if start >= current_end:
            final_segments.append((start, start + 15))  # 15s clips
            current_end = start + 15
            if sum(end-start for start,end in final_segments) >= 55:  # Leave 5s buffer
                break
    
    clips = []
    # Create clips from each non-overlapping segment with effects
    for start, end in final_segments:
        # Create base clip
        clip = video.subclipped(start, min(end, video.end))
        
        # Add zoom effect (gradually zoom in from 1.0 to 1.05 over the clip duration)
        clip = clip.resized(lambda t: 1 + 0.05 * (t/clip.duration))
        
        # Add fade in/out transitions
        fadein = FadeIn(duration=0.5)
        fadeout = FadeOut(duration=0.5)
        clip = fadein.apply(clip)
        clip = fadeout.apply(clip)
        
        clips.append(clip)
    
    # Concatenate all clips
    #TODO add transition clip
    final_clip = clips[0]
    for clip in clips[1:]:
        final_clip = concatenate_videoclips([final_clip, clip], method="compose")
    
    # Convert to vertical (9:16)
    final_clip = final_clip.resized(height=1920)
    final_clip = final_clip.cropped(x_center=final_clip.w/2, y_center=final_clip.h/2, width=1080, height=1920)
    
    final_clip = CompositeVideoClip([final_clip])
    
    final_clip.write_videofile("short_highlights.mp4", fps=24)
    video.close()

if __name__ == "__main__":
    video_url = 'https://www.youtube.com/watch?v=-4GmbBoYQjE'
    download_video(video_url)
    print("Video downloaded successfully.")
    segments = get_transcription()

    # Find highlights
    highlights = find_highlight_segments(segments)

    # Create single short with all highlights
    create_short(highlights)
        