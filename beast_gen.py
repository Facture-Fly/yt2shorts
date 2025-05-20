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
from vidgear.gears import WriteGear
import cv2
import numpy as np
from tqdm import tqdm
import time

amd_render_node = "/dev/dri/renderD128" # <<< IMPORTANT: VERIFY AND UPDATE THIS

ffmpeg_params_amd_fast = [
    # Hardware device initialization (global options)
    "-init_hw_device", "vaapi=va_amd:/dev/dri/renderD128",
    "-filter_hw_device", "va_amd",

    # Hardware encoding parameters
    "-vf", "format=nv12,hwupload",
    "-c:v", "h264_vaapi",
    
    # Quality/speed tradeoff (0-100, lower=better quality)
    "-global_quality", "35",  
    
    # Compression efficiency preset (fast decode vs. quality)
    "-compression_level", "1",  # 1=fastest, 7=best compression
    
    # Parallel frame processing
    "-extra_hw_frames", "4",  # Allow 4 concurrent frames in GPU memory
    
    # B-frame settings for faster encoding
    "-bf", "0",  # Disable B-frames
    
    # GOP structure
    "-g", "250",  # Larger GOP for faster encoding
    "-keyint_min", "250",
    
    # Rate control
    "-rc_mode", "1",  # CQP mode (constant quantization parameter)
]


print("imageio-ffmpeg path:", imageio_ffmpeg.get_ffmpeg_exe())

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
    
    # Configure WriteGear with VAAPI acceleration
    output_params = {
        # Remove problematic parameters
        "-input_framerate": "24",
        
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
    
    writer = WriteGear(
        output="beast_short.mp4",
        compression_mode=True,
        logging=True,
        **output_params
    )

    # Process video segments
    cap = cv2.VideoCapture("input_video.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    def bgr_to_nv12(frame):
        """Convert BGR to properly structured NV12 frame"""
        # Convert to YUV420 first
        yuv420 = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
        
        # Reshape to planar format
        yuv420 = yuv420.reshape((TARGET_HEIGHT * 3 // 2, TARGET_WIDTH))
        
        # Split into Y and UV components
        y_plane = yuv420[:TARGET_HEIGHT, :]
        uv_plane = yuv420[TARGET_HEIGHT:, :].reshape((TARGET_HEIGHT // 2, TARGET_WIDTH // 2, 2))
        
        # Interleave UV components for NV12
        uv_interleaved = uv_plane.reshape((TARGET_HEIGHT // 2, TARGET_WIDTH))
        
        # Combine Y and UV planes
        nv12_frame = np.vstack([y_plane, uv_interleaved])
        
        # Validate final dimensions
        assert nv12_frame.shape == (TARGET_HEIGHT * 3 // 2, TARGET_WIDTH), \
            f"Invalid NV12 shape: {nv12_frame.shape}"
        
        return nv12_frame.astype(np.uint8)
    
    def convert_to_nv12(frame):
        # Convert BGR to NV12 using optimized OpenCV pipeline
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
        yuv = yuv.reshape((TARGET_HEIGHT * 3 // 2, TARGET_WIDTH))
        return yuv

    
    # Mr. Beast effect parameters
    ZOOM_FACTOR = 0.08
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

    def apply_effects(frame, frame_idx):
        # Store original dimensions
        original_height, original_width = frame.shape[:2]
        
        # 1. Zoom with bounded dimensions
        zoom = 1 + ZOOM_FACTOR * np.sin(frame_idx * np.pi/15)
        zoomed = cv2.resize(frame, None, fx=zoom, fy=zoom)
        
        # Crop center to original size after zoom
        zh, zw = zoomed.shape[:2]
        zoom_crop = zoomed[
            max(0, zh//2 - original_height//2):min(zh, zh//2 + original_height//2),
            max(0, zw//2 - original_width//2):min(zw, zw//2 + original_width//2)
        ]
        
        # 2. Face tracking with fixed output size
        cropped = dynamic_crop(zoom_crop)
        
        # 3. Final resize to ensure dimensions
        final_frame = cv2.resize(cropped, (TARGET_WIDTH, TARGET_HEIGHT))
        
        # Add text
        final_frame = cv2.putText(final_frame, **TEXT_SETTINGS)
        
        return final_frame

    def dynamic_crop(frame):
        nonlocal prev_x
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
        
        if faces:
            x, y, fw, fh = max(faces, key=lambda f: f[2]*f[3])
            target_x = x + fw//2
        else:
            target_x = w//2 * (1 + 0.05 * np.sin(time.time()))
        
        # Apply moving average
        position_buffer.append(target_x)
        smoothed_x = np.mean(position_buffer)
        
        # Limit movement speed
        max_delta = 400 * (1/24)  # pixels/frame
        smoothed_x = np.clip(smoothed_x, prev_x - max_delta, prev_x + max_delta)
        prev_x = smoothed_x
        
        # Calculate crop bounds
        left = int(smoothed_x - TARGET_WIDTH//2)
        left = max(0, min(left, w - TARGET_WIDTH))
        right = left + TARGET_WIDTH
        
        # Crop and pad if necessary
        cropped = frame[:, left:right]
        if cropped.shape[1] != TARGET_WIDTH:
            padded = np.zeros((h, TARGET_WIDTH, 3), dtype=np.uint8)
            padded[:, :cropped.shape[1]] = cropped
            return padded
        
        return cropped

    for highlight in highlights:
        start_frame = int(highlight["start"] * fps)
        end_frame = int((highlight["start"] + 15) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break

            # Process and validate frame
            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
            nv12_frame = bgr_to_nv12(frame)
            
            # Write to hardware encoder
            writer.write(nv12_frame)

    # for highlight in tqdm(highlights, desc="Processing highlights"):
    #     start_frame = int(highlight["start"] * fps)
    #     end_frame = int((highlight["start"] + 15) * fps)
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    #     for frame_idx in range(end_frame - start_frame):
    #         ret, frame = cap.read()
    #         if not ret:
    #             break

    #         processed_frame = apply_effects(frame, frame_idx)
    #         writer.write(processed_frame)
    
    cap.release()
    writer.close()


    

# Mr. Beast style video creation
def create_short(highlights):
    video = VideoFileClip("input_video.mp4")
    clips = []
    video_duration = video.duration

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
        "transition_duration": 0.5,  # Increased from 0.3 for smoother blend
        "transition_types": ["fade", "slide", "zoom"],  # Multiple transition options
        "crossfade_duration": 0.3  # Specific duration for crossfades
    }
    
    for i, highlight in enumerate(highlights):
        start = highlight["start"]
        # Inside the loop in create_short
        base_clip_start = start
        base_clip_duration = 15 # Desired duration of the core content of the highlight

        # Determine buffer sizes
        buffer_pre = 1.0 if i > 0 else 0.0 # 1 second buffer before, if not the first clip
        buffer_post = 1.0 if i < len(highlights) - 1 else 0.0 # 1 second buffer after, if not the last clip

        # Calculate actual start and end times from the original video
        actual_subclip_start = max(0, base_clip_start - buffer_pre)
        actual_subclip_end = min(video_duration, base_clip_start + base_clip_duration + buffer_post)

        if actual_subclip_start >= actual_subclip_end:
            print(f"Warning: Skipping highlight at {start}s due to invalid time range after buffering: {actual_subclip_start} to {actual_subclip_end}")
            continue

        clip = video.subclipped(actual_subclip_start, actual_subclip_end)
        

        # Enhanced zoom effect with easing
        clip = clip.resized(lambda t: 1 + BEAST_STYLE["zoom_speed"] * np.sin(t * np.pi/15))
        
        # Add impact text
        txt = TextClip(**BEAST_STYLE["text_style"])
        txt = txt.with_position(("center", "top")).with_duration(2)
        clip = CompositeVideoClip([clip, txt])
        
        if i > 0:
            clip = clip.with_effects([vfx.CrossFadeIn(BEAST_STYLE["crossfade_duration"])])

        clips.append(clip)
    
    # Fast-paced transitions
    slidein = vfx.SlideIn(0.3, "left")
    transition = slidein.apply(clip)

    slided_clips = [
            CompositeVideoClip([c.with_effects([vfx.SlideIn(1, "left")])])
            for c in clips
        ]

    # Add duration validation before concatenation
    total_duration = sum([c.duration for c in clips])
    if total_duration > video_duration:
        print(f"Warning: Total clip duration {total_duration}s exceeds video duration {video_duration}s")
        # Automatically trim last clip
        excess = total_duration - video_duration
        clips[-1] = clips[-1].subclipped(0, clips[-1].duration - excess)

    final = concatenate_videoclips(clips=slided_clips, padding=-BEAST_STYLE["transition_duration"])
    final = final.with_duration(min(final.duration, video_duration))
    # Vertical crop with motion
    final = final.resized(height=1920)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    position_buffer = deque(maxlen=5)
    prev_x = None
    def apply_face_centered_crop(get_frame, t):
        nonlocal prev_x  # Added this line
        frame = get_frame(t)
        height, width = frame.shape[:2]
        
        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
        
        # Calculate target position
        if len(faces) > 0:
            (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])
            target_x = x + w//2
        else:
            # Smooth fallback with sine wave interpolation
            target_x = width/2 * (1 + 0.05 * np.sin(t * 2))
        
        # Apply moving average
        position_buffer.append(target_x)
        smoothed_x = np.mean(position_buffer)
        
        # Limit movement speed
        if prev_x is not None:
            max_delta = MAX_SPEED * (1/24)  # Max movement per frame
            smoothed_x = np.clip(smoothed_x, 
                                prev_x - max_delta, 
                                prev_x + max_delta)
        prev_x = smoothed_x
        
        # Calculate crop
        left = int(smoothed_x - 540)
        left = max(0, min(left, width - 1080))
        
        return frame[:1920, left:left+1080, :3]

    final.write_videofile(
        "beast_short.mp4",
        codec="h264_vaapi",
        fps=24,
        threads=8,  # For CPU-bound pre-processing
        ffmpeg_params=ffmpeg_params_amd_fast,
        logger="bar",
        audio_codec="aac"  # Keep audio separate from GPU processing
    )
    # Replace the existing apply_shake with this
    final = final.transform(apply_face_centered_crop).without_mask()
    w, h = final.size
    fps = final.fps
    duration = final.duration

    # FFmpeg command with correct parameter order
    cmd = [
        "/usr/bin/ffmpeg", "-y",
        "-init_hw_device", "vaapi=va_amd:/dev/dri/renderD128",
        "-filter_hw_device", "va_amd",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-vf", "format=nv12,hwupload",
        "-c:v", "h264_vaapi",
        "-global_quality", "35",
        "-compression_level", "1",
        "-bf", "0",
        "-g", "250",
        "-loglevel", "debug",  # Get detailed encoding stats
        "beast_short.mp4"
    ]
    
    
    # Pipe frames directly to FFmpeg
    # with final.iter_frames(fps=fps, dtype=np.uint8) as frames:
    #     proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        
    #     try:
    #         for frame in frames:
    #             # Convert RGB to NV12 upfront
    #             nv12_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV_I420)
    #             proc.stdin.write(nv12_frame.tobytes())
    #     finally:
    #         proc.stdin.close()
    #         proc.wait()

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

