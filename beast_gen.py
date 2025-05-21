import os
os.environ["FFMPEG_BINARY"] = "/usr/bin/ffmpeg"  # Your system FFmpeg path
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

from pytubefix import YouTube
from pytubefix.exceptions import VideoUnavailable
import whisper
from textblob import TextBlob
from moviepy import * 
import numpy as np
import spacy
import cv2
from collections import deque
import os
import pickle
import hashlib
import re
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from spacy.matcher import PhraseMatcher
from moviepy.config import check
import imageio_ffmpeg
import moviepy.config as mpy_conf
import subprocess
from vidgear.gears import WriteGear, VideoGear
import cv2
import numpy as np
from tqdm import tqdm
import time


ffmpeg_path = "/usr/bin/ffmpeg"
os.environ["FFMPEG_BINARY"] = ffmpeg_path
# Ensure MoviePy uses your FFmpeg binary
check()
MAX_SPEED = 400  # pixels per second
TRANSCRIPTION_CACHE_DIR = ".transcription_cache"
MODEL_VERSION = "base"   # Change this if you switch models

amd_render_node = "/dev/dri/renderD128" # <<< IMPORTANT: VERIFY AND UPDATE THIS


def get_video_checksum(file_path):
    """Create unique hash based on file contents and metadata"""
    file_stat = os.stat(file_path)
    hash_data = f"{file_stat.st_size}-{file_stat.st_mtime_ns}"
    return hashlib.md5(hash_data.encode()).hexdigest()

def get_transcription(force_refresh=False):
    # Create cache directory if needed
    os.makedirs(TRANSCRIPTION_CACHE_DIR, exist_ok=True)
    
    video_file = "input_video.mp4"
    cache_key = f"{get_video_checksum(video_file)}_{MODEL_VERSION}"
    cache_path = os.path.join(TRANSCRIPTION_CACHE_DIR, f"{cache_key}.pkl")
    
    # Return cached result if available
    if not force_refresh and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            print(f"Loading cached transcription from {cache_path}")
            return pickle.load(f)
    
    # Generate new transcription
    model = whisper.load_model(MODEL_VERSION)
    result = model.transcribe(video_file)
    
    # Cache the results
    with open(cache_path, "wb") as f:
        pickle.dump(result["segments"], f)
        print(f"Cached transcription to {cache_path}")
    
    return result["segments"]

# Cache NLP models and sentiment analyzer
nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = SentimentIntensityAnalyzer()

def find_highlight_segments(segments, top_n=5, min_duration=5, max_duration=15, 
                           min_segment_gap=60, positional_boost=True):
    """Enhanced highlight detection for Mr. Beast-style content"""
    # Expanded keyword library with multi-word phrases
    BEAST_KEYWORDS = {
        'challenge': 2.0,
        'million dollars': 3.5,
        'last to leave': 3.2,
        'world record': 3.0,
        '24 hour challenge': 3.5,
        'extreme challenge': 2.8,
        'cash prize': 3.0,
        'brand new': 2.0,
        'subscribe button': 2.5,
        'once in a lifetime': 3.0,
        'life-changing': 3.0,
        'insane challenge': 3.2,
        'huge giveaway': 3.5,
        'custom build': 2.8,
        'survival challenge': 3.0
    }

    # Configure NLP pipeline
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(text) for text in BEAST_KEYWORDS.keys()]
    matcher.add("BEAST_KEYWORDS", patterns)

    # Enhanced entity patterns
    ruler = nlp.add_pipe("entity_ruler")
    entity_patterns = [
        {"label": "MONEY", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["million", "thousand", "billion", "dollars"]}}]},
        {"label": "TIME", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["hours", "days", "minutes"]}}]},
        {"label": "CHALLENGE", "pattern": [{"LOWER": {"IN": ["challenge", "competition", "battle", "showdown"]}}]},
        {"label": "SOCIAL", "pattern": [{"LOWER": "subscribe"}, {"LOWER": "to"}, {"LOWER": "my"}, {"LOWER": "channel"}]}
    ]
    ruler.add_patterns(entity_patterns)

    # Calculate video duration if available
    total_duration = segments[-1]["end"] if segments else 0

    scored_segments = []
    for segment in segments:
        text = segment["text"].lower()
        score = 0
        duration = segment["end"] - segment["start"]
        
        # 1. Phrase-based keyword matching
        doc = nlp(text)
        matches = matcher(doc)
        for match_id, start, end in matches:
            matched_phrase = doc[start:end].text
            score += BEAST_KEYWORDS.get(matched_phrase, 0)

        # 2. Entity recognition boosts
        entities = {ent.label_: list(doc.ents).count(ent) for ent in doc.ents}
        score += entities.get("MONEY", 0) * 3.0
        score += entities.get("TIME", 0) * 2.0
        score += entities.get("CHALLENGE", 0) * 2.5
        score += entities.get("SOCIAL", 0) * 2.0

        # 3. Enhanced sentiment analysis with excitement detection
        sentiment = sentiment_analyzer.polarity_scores(text)
        sentiment_boost = abs(sentiment['compound']) * 2.0
        exclamation_boost = text.count('!') * 0.5
        all_caps_boost = len(re.findall(r'\b[A-Z]{3,}\b', segment["text"])) * 0.3
        score += sentiment_boost + exclamation_boost + all_caps_boost

        # 4. Dynamic duration scoring
        if min_duration <= duration <= max_duration:
            ideal = (min_duration + max_duration) / 2
            duration_score = 1 - (abs(duration - ideal) / (max_duration - min_duration))
            score += duration_score * 1.5

        # 5. Position-based scoring
        if positional_boost and total_duration > 0:
            position = segment["start"] / total_duration
            if segment["start"] < 120:  # First 2 minutes
                score *= 1.4
            elif position > 0.7:  # Last 30%
                score *= 1.3
            elif 0.4 < position < 0.6:  # Middle 20%
                score *= 1.2

        scored_segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "score": score,
            "text": segment["text"]
        })

    # Sort by descending score
    sorted_segments = sorted(scored_segments, key=lambda x: x["score"], reverse=True)

    # Selection with temporal diversity
    final_segments = []
    
    # Prioritize opening hook
    opening_segments = [s for s in sorted_segments if s["start"] < 120]
    if opening_segments:
        final_segments.append(opening_segments[0])
    
    # Select remaining segments with gap enforcement
    for seg in sorted_segments:
        if len(final_segments) >= top_n:
            break
        if seg in final_segments:
            continue
        # Check temporal proximity
        if all(abs(seg["start"] - existing["start"]) >= min_segment_gap
               for existing in final_segments):
            final_segments.append(seg)
    
    # Fallback if not enough segments
    while len(final_segments) < top_n and len(sorted_segments) > 0:
        remaining = [s for s in sorted_segments if s not in final_segments]
        if not remaining:
            break
        final_segments.append(remaining[0])

    return final_segments[:top_n]

def create_vid_gear_short(highlights):
    TARGET_WIDTH = 1080
    TARGET_HEIGHT = 1920

    # Initialize face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Configure WriteGear with optimized settings
    output_params = {
        "-vcodec": "libx264",
        "-preset": "medium",
        "-crf": "23",
        "-pix_fmt": "yuv420p",
        "-threads": "8",
        "-g": "50",
        "-bf": "2"
    }
    
    writer = WriteGear(
        output="beast_short.mp4",
        compression_mode=True,
        logging=True,
        **output_params
    )

    # Process video segments
    cap = cv2.VideoCapture("input_video.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Mr. Beast effect parameters
    ZOOM_FACTOR = 0.08  # Reduced from original
    ZOOM_DURATION = 15   # frames for zoom effect
    TEXT_SETTINGS = {
        "text": "EPIC CHALLENGE",
        "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
        "fontScale": 1.5,
        "color": (255, 255, 255),
        "thickness": 3,
        "lineType": cv2.LINE_AA,
        "org": (540, 100)
    }

    # Face tracking state
    position_buffer = deque(maxlen=5)
    prev_x = TARGET_WIDTH // 2

    def apply_effects(frame, frame_idx, highlight_duration):
        # 1. Initial resize to target dimensions
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        
        # 2. Controlled zoom effect (only at start/end of highlight)
        if frame_idx < ZOOM_DURATION or frame_idx > highlight_duration - ZOOM_DURATION:
            # Smooth zoom factor using cosine curve
            progress = min(frame_idx / ZOOM_DURATION, 1.0) if frame_idx < ZOOM_DURATION else \
                      min((highlight_duration - frame_idx) / ZOOM_DURATION, 1.0)
            
            zoom = 1 + ZOOM_FACTOR * (1 - np.cos(progress * np.pi)) / 2
            frame = cv2.resize(frame, None, fx=zoom, fy=zoom)
            
            # Recenter after zoom
            h, w = frame.shape[:2]
            frame = frame[h//2 - TARGET_HEIGHT//2:h//2 + TARGET_HEIGHT//2,
                          w//2 - TARGET_WIDTH//2:w//2 + TARGET_WIDTH//2]
        
        # 3. Apply dynamic face tracking crop
        final_frame = dynamic_crop(frame)
        
        # 4. Add text overlay
        final_frame = cv2.putText(final_frame, **TEXT_SETTINGS)
        
        return final_frame

    def dynamic_crop(frame):
        nonlocal prev_x
        h, w = frame.shape[:2]
        
        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
        
        # Calculate target position
        if len(faces) > 0:
            areas = faces[:, 2] * faces[:, 3]
            largest_idx = np.argmax(areas)
            (x, y, fw, fh) = faces[largest_idx]
            target_x = x + fw//2
        else:
            target_x = w//2  # Default to center
        
        # Smooth movement
        position_buffer.append(target_x)
        smoothed_x = np.mean(position_buffer)
        
        # Limit movement speed
        max_delta = 400 * (1/24)
        smoothed_x = np.clip(smoothed_x, prev_x - max_delta, prev_x + max_delta)
        prev_x = smoothed_x
        
        # Calculate crop bounds
        left = int(smoothed_x - TARGET_WIDTH//2)
        left = max(0, min(left, w - TARGET_WIDTH))
        
        return frame[:, left:left + TARGET_WIDTH]

    for highlight in tqdm(highlights, desc="Processing highlights"):
        start_frame = int(highlight["start"] * fps)
        end_frame = int((highlight["start"] + 15) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        highlight_duration = end_frame - start_frame

        for frame_idx in range(highlight_duration):
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = apply_effects(frame, frame_idx, highlight_duration)
            writer.write(processed_frame)

    cap.release()
    writer.close()


def create_short(highlights):
    # Open video using VideoGear instead of MovieFileClip
    stream = VideoGear(source="input_video.mp4").start()
    
    # Get video info using OpenCV to get duration and specs
    cap = cv2.VideoCapture("input_video.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = frame_count / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Mr. Beast-style effects configuration
    BEAST_STYLE = {
        "zoom_speed": 0.08,
        "text_style": {
            "text": "EPIC CHALLENGE",
            "font_size": 45,
            "color": (255, 255, 255),  # White in BGR
            "font": cv2.FONT_HERSHEY_DUPLEX,  # OpenCV font
            "stroke_color": (0, 0, 0),  # Black in BGR
            "stroke_width": 3
        },
        "transition_duration": 0.5,
        "crossfade_duration": 0.3
    }
    
    # Create WriteGear instance with appropriate parameters
    # Modified to fix VAAPI issues
    output_params = {
        # Remove problematic parameters
        "-input_framerate": fps,
        
        # Use CPU encoding instead of VAAPI since there are compatibility issues
        "-vcodec": "libx264",
        "-preset": "medium",
        "-crf": "23",  # Quality setting (lower is better)
        
        # Audio settings
        "-ac": "2",
        "-ar": "44100",
        "-acodec": "aac",
        
        # Other optimization settings
        "-pix_fmt": "yuv420p",  # Standard pixel format
        "-threads": "8",
        "-g": "50",  # Keyframe interval
        "-bf": "2"   # B-frames
    }
    
    writer = WriteGear(output="beast_short.mp4", compression_mode=True, logging=True, **output_params)
    
    # Setup face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    position_buffer = deque(maxlen=5)
    prev_x = None
    MAX_SPEED = 10  # Defined MAX_SPEED which was missing in original code
    
    # Process each highlight segment
    for i, highlight in enumerate(highlights):
        start = highlight["start"]
        base_clip_duration = 15  # Desired duration of the core content
        
        # Determine buffer sizes
        buffer_pre = 1.0 if i > 0 else 0.0
        buffer_post = 1.0 if i < len(highlights) - 1 else 0.0
        
        # Calculate actual start and end times
        actual_subclip_start = max(0, start - buffer_pre)
        actual_subclip_end = min(video_duration, start + base_clip_duration + buffer_post)
        
        if actual_subclip_start >= actual_subclip_end:
            print(f"Warning: Skipping highlight at {start}s due to invalid time range after buffering")
            continue
        
        # Calculate frame positions
        start_frame = int(actual_subclip_start * fps)
        end_frame = int(actual_subclip_end * fps)
        total_frames = end_frame - start_frame
        
        # Set the position in the video using OpenCV
        cap = cv2.VideoCapture("input_video.mp4")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Process frames for this clip
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_idx / fps
            
            # Apply zoom effect
            zoom_factor = 1 + BEAST_STYLE["zoom_speed"] * np.sin(current_time * np.pi/15)
            if zoom_factor != 1.0:
                h, w = frame.shape[:2]
                center_x, center_y = w//2, h//2
                M = cv2.getRotationMatrix2D((center_x, center_y), 0, zoom_factor)
                frame = cv2.warpAffine(frame, M, (w, h))
            
            # Add text overlay for first 2 seconds
            if current_time < 2.0:
                # Create text overlay
                text = BEAST_STYLE["text_style"]["text"]
                font = BEAST_STYLE["text_style"]["font"]
                font_size = BEAST_STYLE["text_style"]["font_size"] / 30  # Adjust for OpenCV scale
                color = BEAST_STYLE["text_style"]["color"]
                stroke_color = BEAST_STYLE["text_style"]["stroke_color"]
                stroke_width = BEAST_STYLE["text_style"]["stroke_width"]
                
                # Get text size
                text_size = cv2.getTextSize(text, font, font_size, stroke_width)[0]
                text_x = (frame.shape[1] - text_size[0]) // 2
                text_y = 50 + text_size[1]  # Position at top with some padding
                
                # Draw text with stroke (outline)
                cv2.putText(frame, text, (text_x, text_y), font, font_size, 
                           stroke_color, stroke_width + 2, cv2.LINE_AA)
                cv2.putText(frame, text, (text_x, text_y), font, font_size, 
                           color, stroke_width, cv2.LINE_AA)
            
            # Apply face-centered crop
            if frame is not None and frame.size > 0:  # Make sure frame is valid
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
                
                # Calculate target position
                h, w = frame.shape[:2]
                if len(faces) > 0:
                    (x, y, w_face, h_face) = max(faces, key=lambda f: f[2]*f[3])
                    target_x = x + w_face//2
                else:
                    # Smooth fallback with sine wave interpolation
                    target_x = w/2 * (1 + 0.05 * np.sin(current_time * 2))
                
                # Apply moving average
                position_buffer.append(target_x)
                smoothed_x = np.mean(position_buffer)
                
                # Limit movement speed
                if prev_x is not None:
                    max_delta = MAX_SPEED * (1/fps)  # Max movement per frame
                    smoothed_x = np.clip(smoothed_x, 
                                      prev_x - max_delta, 
                                      prev_x + max_delta)
                prev_x = smoothed_x
                
                # Calculate crop
                left = int(smoothed_x - 540)
                left = max(0, min(left, w - 1080))
                
                # Ensure safe cropping bounds
                if left + 1080 <= w:
                    # Crop and resize to vertical format (1080x1920)
                    cropped = frame[:, left:left+1080]
                    if cropped.shape[0] != 1920:  # If height isn't already 1920
                        frame = cv2.resize(cropped, (1080, 1920))
                    else:
                        frame = cropped
                else:
                    # Not enough width to crop, resize the whole frame instead
                    frame = cv2.resize(frame, (1080, 1920))
                
                # Apply slide-in transition effect for first few frames
                if i > 0 and frame_idx < int(BEAST_STYLE["transition_duration"] * fps):
                    transition_progress = frame_idx / (BEAST_STYLE["transition_duration"] * fps)
                    offset = int((1 - transition_progress) * w)
                    blank = np.zeros_like(frame)
                    
                    # Ensure correct dimensions for hstack
                    if offset < frame.shape[1]:
                        # Create transition effect
                        combined = np.hstack([blank[:, :offset], frame[:, offset:]])
                        if combined.shape[1] > 1080:
                            combined = combined[:, :1080]
                        frame = combined
                
                # Write the processed frame
                writer.write(frame)
        
        cap.release()
    
    # Close resources
    stream.stop()
    writer.close()
    
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

# video_url="https://www.youtube.com/watch?v=zajUgQLviwk"
# download_video(video_url)
# print("Video downloaded successfully.")
# segments = get_transcription()
# print(segments)
# print("Segmented successfully successfully.")
# highlights = find_highlight_segments(segments)
# create_short(highlights)

