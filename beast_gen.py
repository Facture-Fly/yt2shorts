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
from moviepy.video.fx import Crop

def get_transcription():
    model = whisper.load_model("base")  # Use 'small' or 'medium' for better accuracy
    result = model.transcribe("input_video.mp4")
    return result["segments"]

# Custom Mr. Beast analyzer
def find_highlight_segments(segments, top_n=5, min_duration=5, max_duration=15):
    """Optimized for Mr. Beast video structure"""
    # Mr. Beast-specific keywords
    BEAST_KEYWORDS = {
        'challenge': 2.0,
        'million': 3.0,
        'dollars': 3.0,
        'giveaway': 2.5,
        'last to leave': 3.0,
        'win': 2.0,
        'destroy': 1.8,
        '24 hours': 2.2,
        'expensive': 2.0,
        'custom': 1.5,
        'world record': 2.5,
        'free': 2.0,
        'vs': 1.8,
        'winning': 2.0,
        'prize': 2.2
    }

    # Load custom patterns
    nlp = spacy.load("en_core_web_sm")
    ruler = nlp.add_pipe("entity_ruler")
    patterns = [
        {"label": "MONEY", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["million", "thousand", "billion"]}}]},
        {"label": "TIME", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["hours", "days", "minutes"]}}]}
    ]

    ruler.add_patterns(patterns)

    scored_segments = []
    
    for segment in segments:
        text = segment["text"].lower()
        score = 0
        
        # Base score from keyword matching
        for keyword, weight in BEAST_KEYWORDS.items():
            if keyword in text:
                score += weight * text.count(keyword)
        
        # Analyze with spaCy
        doc = nlp(segment["text"])
        
        # Boost for money/time entities
        money_ents = [ent for ent in doc.ents if ent.label_ == "MONEY"]
        time_ents = [ent for ent in doc.ents if ent.label_ == "TIME"]
        score += len(money_ents) * 2.5
        score += len(time_ents) * 1.5
        
        # Sentiment analysis for emotional payoff moments
        analysis = TextBlob(text)
        sentiment_boost = abs(analysis.sentiment.polarity) * 1.5
        if analysis.sentiment.polarity > 0.7:  # Extreme positive = giveaway moment
            sentiment_boost *= 2
           
        score += sentiment_boost
        
        # Duration scoring (ideal 7-12 seconds)
        duration = segment["end"] - segment["start"]
        duration_score = 1 - abs(duration - 10)/10  # Peak at 10 seconds
        score += duration_score * 1.2
        
        scored_segments.append((segment["start"], score))
    
    # Mr. Beast structure rules
    final_segments = []
    sorted_segments = sorted(scored_segments, key=lambda x: x[1], reverse=True)
    
    # Prioritize opening hook (first 2 minutes)
    opening_segments = [s for s in sorted_segments if s[0] < 120]
    if opening_segments:
        final_segments.append(opening_segments[0])
    
    # Then best remaining segments
    remaining = [s for s in sorted_segments if s not in final_segments]
    final_segments += remaining[:top_n-1]
    
    return final_segments[:top_n]




# Mr. Beast style video creation
def create_short(highlights):
    video = VideoFileClip("input_video.mp4")
    clips = []
    
    # Mr. Beast-style effects
    BEAST_STYLE = {
        "zoom_speed": 0.08,
        "text_style": {
            "text": "EPIC CHALLENGE",
            "font_size": 45,
            "color": "white",
            "font": "/usr/share/fonts/opentype/fira/FiraMono-Medium.otf",
            "stroke_color": "black",
            "stroke_width": 3
        },
        "transition_duration": 0.3
    }
    print(BEAST_STYLE["text_style"])
    
    for start, _ in highlights:
        clip = video.subclipped(start, start + 15)
        
        # Dynamic zoom effect
        clip = clip.resized(lambda t: 1 + BEAST_STYLE["zoom_speed"] * t)
        
        # Add impact text
        txt = TextClip(**BEAST_STYLE["text_style"])
        txt = txt.with_position(("center", "top")).with_duration(2)
        clip = CompositeVideoClip([clip, txt])
        
        
        clips.append(clip)
    
    # Fast-paced transitions
    slidein = vfx.SlideIn(0.3, "left")
    transition = slidein.apply(clip)

    slided_clips = [
            CompositeVideoClip([clip.with_effects([vfx.SlideIn(1, "left")])])
            for clip in clips
        ]
    final = concatenate_videoclips(clips=slided_clips, padding=-BEAST_STYLE["transition_duration"])

    # Vertical crop with motion
    final = final.resized(height=1920)

    # def shaking_position(t):
    #     return final.w/2 * (1 + 0.05 * np.sin(t * 2))
    
    # final = final.fx(
    #     Crop,
    #     x_center=shaking_position,
    #     y_center=final.h/2,
    #     width=1080,
    #     height=1920
    # )
    def apply_shake(get_frame, t):
        frame = get_frame(t)
        
        # Calculate dynamic position
        x_center = int(final.w/2 * (1 + 0.05 * np.sin(t * 2)))
        y_center = final.h // 2
        
        # Ensure crop dimensions
        left = max(0, x_center - 540)
        right = left + 1080  # Force exact width
        
        # Handle right edge
        if right > final.w:
            right = final.w
            left = right - 1080
        
        # Remove alpha channel if present
        return frame[:1920, left:right, :3]

    # Apply transformation and remove mask
    final = final.transform(apply_shake).without_mask()
    
    final.write_videofile("beast_short.mp4", fps=24, threads=4)

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

video_url="https://www.youtube.com/watch?v=zajUgQLviwk"
download_video(video_url)
print("Video downloaded successfully.")
segments = get_transcription()
print(segments)
print("Segmented successfully successfully.")
highlights = find_highlight_segments(segments)
create_short(highlights)

