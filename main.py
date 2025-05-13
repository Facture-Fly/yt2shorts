from pytubefix import YouTube
from pytubefix.exceptions import VideoUnavailable
import whisper
from textblob import TextBlob
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
import os

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

def find_highlight_segments(segments, top_n=3):
    scored_segments = []
    
    for segment in segments:
        text = segment["text"]
        analysis = TextBlob(text)
        # Score based on sentiment and length
        score = analysis.sentiment.polarity * len(text.split())
        scored_segments.append((segment["start"], score))
    
    # Return top N segments by score
    return sorted(scored_segments, key=lambda x: x[1], reverse=True)[:top_n]

def create_short(start_time, duration=60):
    video = VideoFileClip("input_video.mp4")
    
    # Extract clip (max 60s for YouTube Shorts)
    clip = video.subclipped(start_time, min(start_time + duration, video.end))
    
    # Convert to vertical (9:16)
    clip = clip.resized(height=1920)
    clip = clip.cropped(x_center=clip.w/2, y_center=clip.h/2, width=1080, height=1920)
    
    # Add basic text (optional)
    txt_clip = TextClip(font="/usr/share/fonts/opentype/fira/FiraMono-Medium.otf", 
                        text="Highlight!", 
                        font_size=70, 
                        color='white', 
                        bg_color='white')
    final_clip = CompositeVideoClip([clip, txt_clip.with_duration(clip.duration).with_position('center')])
    
    final_clip.write_videofile(f"short_{start_time}.mp4", fps=24)

if __name__ == "__main__":
    video_url = 'https://www.youtube.com/watch?v=-4GmbBoYQjE'
    #download_video(video_url)
    print("Video downloaded successfully.")
    segments = get_transcription()

    # Find highlights
    highlights = find_highlight_segments(segments)

    # Create shorts
    for start_time, score in highlights:
        create_short(start_time)
        