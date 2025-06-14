import os

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
from faster_whisper import WhisperModel
import torch
import yt_dlp

print(f"CUDA (ROCm) available: {torch.cuda.is_available()}")
print(f"Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")


# Ensure MoviePy uses your FFmpeg binary
check()
TRANSCRIPTION_CACHE_DIR = ".transcription_cache"

amd_render_node = "/dev/dri/renderD128" # <<< IMPORTANT: VERIFY AND UPDATE THIS


def get_video_checksum(file_path):
    """Create unique hash based on file contents and metadata"""
    file_stat = os.stat(file_path)
    hash_data = f"{file_stat.st_size}-{file_stat.st_mtime_ns}"
    return hashlib.md5(hash_data.encode()).hexdigest()

def get_transcription(force_refresh=False, model_version="base", input_file=None):
    # Create cache directory if needed
    os.makedirs(TRANSCRIPTION_CACHE_DIR, exist_ok=True)
    
    cache_key = f"{get_video_checksum(input_file)}_{model_version}"
    cache_path = os.path.join(TRANSCRIPTION_CACHE_DIR, f"{cache_key}.pkl")
    
    # Return cached result if available
    if not force_refresh and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            print(f"Loading cached transcription from {cache_path}")
            return pickle.load(f)
    
    # Generate new transcription
    #model = whisper.load_model(model_version, device="cuda")
    #TODO fix cuda using cpu for now
    model = WhisperModel(model_version)
    segments, info = model.transcribe(input_file)
    
    segments_list = list(segments)
    # Cache the results
    with open(cache_path, "wb") as f:
        pickle.dump(segments_list, f) 
        print(f"Cached transcription to {cache_path}")
    
    return segments_list

# Cache NLP models and sentiment analyzer
nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = SentimentIntensityAnalyzer()

def find_highlight_segments(segments, top_n=5, min_duration=5, max_duration=15, 
                           min_segment_gap=60, positional_boost=True):
    """Enhanced highlight detection for Mr. Beast-style content"""
    if not isinstance(segments, list):
        segments = list(segments)
    
    # Convert to dictionary format for compatibility
    segments = [
        {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text
        }
        for seg in segments
    ]
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

def create_vid_gear_short(highlights, **params):
    """Create short video with customizable parameters"""
    # Extract parameters with defaults
    TARGET_WIDTH = params.get('output_width', 1080)
    TARGET_HEIGHT = params.get('output_height', 1920)
    ZOOM_FACTOR = params.get('zoom_factor', 0.08)
    ZOOM_DURATION = params.get('zoom_duration', 15)
    MAX_TRACKING_SPEED = params.get('max_tracking_speed', 400)
    SMOOTHING_FRAMES = params.get('smoothing_frames', 5)
    ENABLE_FACE_TRACKING = params.get('enable_face_tracking', True)
    TEXT_LINES = params.get('text_lines', [])
    TEXT_POSITION = params.get('text_position', 'Top')
    TEXT_BG_OPACITY = params.get('text_bg_opacity', 200)
    STROKE_THICKNESS = params.get('stroke_thickness', 3)
    VIDEO_PRESET = params.get('video_preset', 'medium')
    CRF_VALUE = params.get('crf_value', 23)

    # Initialize face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Configure WriteGear with optimized settings
    output_params = {
        "-vcodec": "libx264",
        "-preset": VIDEO_PRESET,
        "-crf": str(CRF_VALUE),
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
    
    # Face tracking state
    position_buffer = deque(maxlen=SMOOTHING_FRAMES)
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
        
        # 3. Apply dynamic face tracking crop if enabled
        if ENABLE_FACE_TRACKING:
            final_frame = dynamic_crop(frame)
        else:
            final_frame = frame
        
        # 4. Add text overlay if configured
        if TEXT_LINES:
            final_frame = add_text_with_background(final_frame)
        
        return final_frame

    def add_text_with_background(frame):
        # Calculate total text dimensions
        text_sizes = []
        for cfg in TEXT_LINES:
            if not cfg.get('text'):
                continue
            (text_width, text_height), _ = cv2.getTextSize(
                cfg["text"], 
                cv2.FONT_HERSHEY_SIMPLEX, 
                cfg.get("fontScale", 1.5), 
                STROKE_THICKNESS
            )
            text_sizes.append((text_width, text_height))

        if not text_sizes:
            return frame

        max_width = max([w for w, h in text_sizes])
        total_height = sum([h for w, h in text_sizes]) + 40 * (len(text_sizes) - 1)

        # Calculate background position based on TEXT_POSITION
        padding = 30
        bg_x1 = (TARGET_WIDTH - max_width) // 2 - padding
        
        if TEXT_POSITION == "Top":
            bg_y1 = 80
        elif TEXT_POSITION == "Center":
            bg_y1 = (TARGET_HEIGHT - total_height) // 2 - padding
        else:  # Bottom
            bg_y1 = TARGET_HEIGHT - total_height - 80 - padding
            
        bg_x2 = bg_x1 + max_width + 2*padding
        bg_y2 = bg_y1 + total_height + padding

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), 
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, TEXT_BG_OPACITY/255, 
                               frame, 1 - TEXT_BG_OPACITY/255, 0)

        # Draw text with stroke effect
        y_position = bg_y1 + padding + text_sizes[0][1]
        for i, cfg in enumerate(TEXT_LINES):
            if not cfg.get('text'):
                continue
                
            # Calculate text position
            (text_width, text_height), _ = cv2.getTextSize(
                cfg["text"], cv2.FONT_HERSHEY_SIMPLEX, 
                cfg.get("fontScale", 1.5), STROKE_THICKNESS
            )
            x = (TARGET_WIDTH - text_width) // 2
            y = y_position + (40 * i)

            # Draw stroke (8 directions)
            for dx in [-2, 0, 2]:
                for dy in [-2, 0, 2]:
                    if dx == 0 and dy == 0:
                        continue
                    frame = cv2.putText(
                        frame, cfg["text"], (x + dx, y + dy),
                        cv2.FONT_HERSHEY_SIMPLEX, cfg.get("fontScale", 1.5),
                        (0, 0, 0), STROKE_THICKNESS + 2,
                        cv2.LINE_AA
                    )

            # Draw main text
            frame = cv2.putText(
                frame, cfg["text"], (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, cfg.get("fontScale", 1.5),
                (255, 255, 255), STROKE_THICKNESS,
                cv2.LINE_AA
            )

        return frame
    
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
        max_delta = MAX_TRACKING_SPEED * (1/fps)
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

def download_video(url):
    ydl_opts = {
        'format': 'best[height<=1080]',  # Get best quality up to 1080p
        'outtmpl': 'input_video.%(ext)s',
        'merge_output_format': 'mp4',
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("Download successful with yt-dlp!")
        
        # Rename to expected filename
        for file in os.listdir('.'):
            if file.startswith('input_video.') and not file.endswith('.mp4'):
                os.rename(file, 'input_video.mp4')
                break
                
    except Exception as e:
        print(f"Error: {str(e)}")
