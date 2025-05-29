import cv2
import numpy as np
from collections import deque
from tqdm import tqdm
from vidgear.gears import WriteGear
import os

def create_vid_gear_short(highlights, input_video_path, output_video_path=None, **params):
    """
    Create short video with customizable parameters from extracted segments
    
    Args:
        highlights: List of highlight segments from ViralShortsCreator
        input_video_path: Path to the input video file
        output_video_path: Path for output video (optional)
        **params: Customizable parameters for video processing
    """
    # Validate inputs
    if not highlights:
        print("No highlights provided")
        return False
    
    if not os.path.exists(input_video_path):
        print(f"Input video not found: {input_video_path}")
        return False
    
    # Set default output path if not provided
    if not output_video_path:
        base_name = os.path.splitext(os.path.basename(input_video_path))[0]
        output_video_path = f"{base_name}_shorts.mp4"
    
    # Extract parameters with defaults
    TARGET_WIDTH = params.get('output_width', 1080)
    TARGET_HEIGHT = params.get('output_height', 1920)
    ZOOM_FACTOR = params.get('zoom_factor', 0.08)
    ZOOM_DURATION = params.get('zoom_duration', 15)
    MAX_TRACKING_SPEED = params.get('max_tracking_speed', 400)
    SMOOTHING_FRAMES = params.get('smoothing_frames', 5)
    ENABLE_FACE_TRACKING = params.get('enable_face_tracking', True)
    TEXT_LINES = params.get('text_lines', [])
    TEXT_POSITION = params.get('text_position', 'Top')  # Top, Center, Bottom
    TEXT_BG_OPACITY = params.get('text_bg_opacity', 200)
    STROKE_THICKNESS = params.get('stroke_thickness', 3)
    VIDEO_PRESET = params.get('video_preset', 'medium')
    CRF_VALUE = params.get('crf_value', 23)
    MAX_SEGMENT_DURATION = params.get('max_segment_duration', 15)  # seconds
    
    print(f"Creating shorts from {len(highlights)} segments...")
    print(f"Output: {output_video_path}")
    
    # Initialize face detection
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        if face_cascade.empty():
            print("Warning: Face cascade not loaded, disabling face tracking")
            ENABLE_FACE_TRACKING = False
    except Exception as e:
        print(f"Error loading face cascade: {e}")
        ENABLE_FACE_TRACKING = False
    
    # Configure WriteGear with optimized settings
    output_params = {
        "-vcodec": "libx264",
        "-preset": VIDEO_PRESET,
        "-crf": str(CRF_VALUE),
        "-pix_fmt": "yuv420p",
        "-threads": "8",
        "-g": "50",
        "-bf": "2",
        "-movflags": "+faststart"  # Enable streaming
    }
    
    try:
        writer = WriteGear(
            output=output_video_path,
            compression_mode=True,
            logging=False,  # Reduce console spam
            **output_params
        )
    except Exception as e:
        print(f"Error initializing video writer: {e}")
        return False

    # Open video capture
    try:
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Error opening video: {input_video_path}")
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        
        print(f"Video info: {fps:.1f} FPS, {video_duration:.1f}s duration")
        
    except Exception as e:
        print(f"Error reading video info: {e}")
        return False
    
    # Face tracking state
    position_buffer = deque(maxlen=SMOOTHING_FRAMES)
    prev_x = TARGET_WIDTH // 2

    def apply_effects(frame, frame_idx, highlight_duration, segment_info=None):
        """Apply visual effects to frame"""
        try:
            # 1. Initial resize to target dimensions
            original_h, original_w = frame.shape[:2]
            
            # Calculate aspect ratio preserving resize
            aspect_ratio = original_w / original_h
            target_aspect = TARGET_WIDTH / TARGET_HEIGHT
            
            if aspect_ratio > target_aspect:
                # Video is wider - fit to height
                new_height = TARGET_HEIGHT
                new_width = int(new_height * aspect_ratio)
            else:
                # Video is taller - fit to width
                new_width = TARGET_WIDTH
                new_height = int(new_width / aspect_ratio)
            
            frame = cv2.resize(frame, (new_width, new_height))
            
            # 2. Controlled zoom effect (only at start/end of highlight)
            if frame_idx < ZOOM_DURATION or frame_idx > highlight_duration - ZOOM_DURATION:
                # Smooth zoom factor using cosine curve
                if frame_idx < ZOOM_DURATION:
                    progress = frame_idx / ZOOM_DURATION
                else:
                    progress = (highlight_duration - frame_idx) / ZOOM_DURATION
                
                zoom = 1 + ZOOM_FACTOR * (1 - np.cos(progress * np.pi)) / 2
                zoomed_width = int(new_width * zoom)
                zoomed_height = int(new_height * zoom)
                
                if zoomed_width > 0 and zoomed_height > 0:
                    frame = cv2.resize(frame, (zoomed_width, zoomed_height))
                    
                    # Recenter after zoom
                    h, w = frame.shape[:2]
                    start_y = max(0, h//2 - TARGET_HEIGHT//2)
                    end_y = min(h, start_y + TARGET_HEIGHT)
                    start_x = max(0, w//2 - TARGET_WIDTH//2)
                    end_x = min(w, start_x + TARGET_WIDTH)
                    
                    frame = frame[start_y:end_y, start_x:end_x]
            
            # 3. Apply dynamic face tracking crop if enabled
            if ENABLE_FACE_TRACKING and frame.shape[1] > TARGET_WIDTH:
                final_frame = dynamic_crop(frame)
            else:
                # Center crop if needed
                h, w = frame.shape[:2]
                if w > TARGET_WIDTH:
                    start_x = (w - TARGET_WIDTH) // 2
                    frame = frame[:, start_x:start_x + TARGET_WIDTH]
                if h > TARGET_HEIGHT:
                    start_y = (h - TARGET_HEIGHT) // 2
                    frame = frame[start_y:start_y + TARGET_HEIGHT, :]
                final_frame = frame
            
            # Ensure exact target dimensions
            if final_frame.shape[:2] != (TARGET_HEIGHT, TARGET_WIDTH):
                final_frame = cv2.resize(final_frame, (TARGET_WIDTH, TARGET_HEIGHT))
            
            # 4. Add text overlay if configured
            if TEXT_LINES or (segment_info and segment_info.get('highlight_text')):
                final_frame = add_text_with_background(final_frame, segment_info)
            
            return final_frame
            
        except Exception as e:
            print(f"Error applying effects: {e}")
            # Return resized frame as fallback
            return cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

    def add_text_with_background(frame, segment_info=None):
        """Add text overlay with background"""
        try:
            # Use provided text lines or extract from segment
            text_lines = TEXT_LINES.copy()
            
            # Add segment text if available and no custom text provided
            if not text_lines and segment_info and segment_info.get('highlight_text'):
                # Split long text into multiple lines
                text = segment_info['highlight_text']
                words = text.split()
                lines = []
                current_line = []
                
                for word in words:
                    current_line.append(word)
                    test_line = ' '.join(current_line)
                    if len(test_line) > 40:  # Max characters per line
                        if len(current_line) > 1:
                            lines.append(' '.join(current_line[:-1]))
                            current_line = [word]
                        else:
                            lines.append(test_line)
                            current_line = []
                
                if current_line:
                    lines.append(' '.join(current_line))
                
                # Convert to text config format
                for line in lines[:3]:  # Max 3 lines
                    text_lines.append({
                        'text': line,
                        'fontScale': 1.2,
                        'color': (255, 255, 255)
                    })
            
            if not text_lines:
                return frame
            
            # Calculate text dimensions
            text_sizes = []
            for cfg in text_lines:
                if not cfg.get('text'):
                    continue
                (text_width, text_height), _ = cv2.getTextSize(
                    cfg["text"], 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    cfg.get("fontScale", 1.2), 
                    STROKE_THICKNESS
                )
                text_sizes.append((text_width, text_height))

            if not text_sizes:
                return frame

            max_width = max([w for w, h in text_sizes])
            total_height = sum([h for w, h in text_sizes]) + 20 * (len(text_sizes) - 1)

            # Calculate background position
            padding = 30
            bg_x1 = max(10, (TARGET_WIDTH - max_width) // 2 - padding)
            
            if TEXT_POSITION == "Top":
                bg_y1 = 80
            elif TEXT_POSITION == "Center":
                bg_y1 = (TARGET_HEIGHT - total_height) // 2 - padding
            else:  # Bottom
                bg_y1 = TARGET_HEIGHT - total_height - 80 - padding
                
            bg_x2 = min(TARGET_WIDTH - 10, bg_x1 + max_width + 2*padding)
            bg_y2 = bg_y1 + total_height + padding

            # Draw semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), 
                         (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, TEXT_BG_OPACITY/255, 
                                   frame, 1 - TEXT_BG_OPACITY/255, 0)

            # Draw text with stroke effect
            y_position = bg_y1 + padding + text_sizes[0][1] if text_sizes else bg_y1 + padding
            for i, cfg in enumerate(text_lines):
                if not cfg.get('text'):
                    continue
                    
                # Calculate text position
                (text_width, text_height), _ = cv2.getTextSize(
                    cfg["text"], cv2.FONT_HERSHEY_SIMPLEX, 
                    cfg.get("fontScale", 1.2), STROKE_THICKNESS
                )
                x = max(bg_x1 + padding, (TARGET_WIDTH - text_width) // 2)
                y = y_position + (30 * i)

                # Draw stroke (black outline)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        cv2.putText(
                            frame, cfg["text"], (x + dx, y + dy),
                            cv2.FONT_HERSHEY_SIMPLEX, cfg.get("fontScale", 1.2),
                            (0, 0, 0), STROKE_THICKNESS + 1,
                            cv2.LINE_AA
                        )

                # Draw main text
                cv2.putText(
                    frame, cfg["text"], (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, cfg.get("fontScale", 1.2),
                    cfg.get("color", (255, 255, 255)), STROKE_THICKNESS,
                    cv2.LINE_AA
                )

            return frame
            
        except Exception as e:
            print(f"Error adding text: {e}")
            return frame
    
    def dynamic_crop(frame):
        """Apply dynamic face tracking crop"""
        nonlocal prev_x
        
        try:
            h, w = frame.shape[:2]
            
            if w <= TARGET_WIDTH:
                return frame
            
            # Face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            
            # Calculate target position
            if len(faces) > 0:
                # Find largest face
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
            if fps > 0:
                max_delta = MAX_TRACKING_SPEED * (1/fps)
                smoothed_x = np.clip(smoothed_x, prev_x - max_delta, prev_x + max_delta)
            prev_x = smoothed_x
            
            # Calculate crop bounds
            left = int(smoothed_x - TARGET_WIDTH//2)
            left = max(0, min(left, w - TARGET_WIDTH))
            
            return frame[:, left:left + TARGET_WIDTH]
            
        except Exception as e:
            print(f"Error in dynamic crop: {e}")
            # Fallback to center crop
            start_x = (frame.shape[1] - TARGET_WIDTH) // 2
            return frame[:, start_x:start_x + TARGET_WIDTH]

    # Process each highlight segment
    try:
        for i, highlight in enumerate(tqdm(highlights, desc="Processing highlights")):
            print(f"\nProcessing segment {i+1}/{len(highlights)}")
            
            # Extract timing information
            start_time = highlight.get("start", 0)
            end_time = highlight.get("end", start_time + MAX_SEGMENT_DURATION)
            duration = min(end_time - start_time, MAX_SEGMENT_DURATION)
            
            # Validate segment timing
            if start_time >= video_duration:
                print(f"Skipping segment {i+1}: start time {start_time}s exceeds video duration")
                continue
            
            if duration <= 0:
                print(f"Skipping segment {i+1}: invalid duration {duration}s")
                continue
            
            print(f"  Time: {start_time:.1f}s - {end_time:.1f}s (duration: {duration:.1f}s)")
            
            # Calculate frame range
            start_frame = int(start_time * fps)
            end_frame = int((start_time + duration) * fps)
            highlight_duration = end_frame - start_frame
            
            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Reset face tracking for each segment
            position_buffer.clear()
            prev_x = TARGET_WIDTH // 2
            
            # Process frames
            frames_processed = 0
            for frame_idx in range(highlight_duration):
                ret, frame = cap.read()
                if not ret:
                    print(f"  Warning: Could not read frame {frame_idx}")
                    break

                try:
                    processed_frame = apply_effects(frame, frame_idx, highlight_duration, highlight)
                    writer.write(processed_frame)
                    frames_processed += 1
                except Exception as e:
                    print(f"  Error processing frame {frame_idx}: {e}")
                    # Write original frame as fallback
                    fallback_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
                    writer.write(fallback_frame)
            
            print(f"  Processed {frames_processed} frames")

    except Exception as e:
        print(f"Error during processing: {e}")
        return False
    
    finally:
        # Cleanup
        cap.release()
        writer.close()
        print(f"\nShort video created: {output_video_path}")
    
    return True


def create_shorts_from_viral_segments(viral_segments, input_video_path, output_dir="shorts", **params):
    """
    Create multiple short videos from viral segments
    
    Args:
        viral_segments: Output from ViralShortsCreator.create_shorts()
        input_video_path: Path to input video
        output_dir: Directory to save shorts
        **params: Video processing parameters
    """
    if not viral_segments:
        print("No viral segments provided")
        return []
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    created_videos = []
    base_name = os.path.splitext(os.path.basename(input_video_path))[0]
    
    for i, segment in enumerate(viral_segments):
        print(f"\n{'='*50}")
        print(f"Creating Short {i+1}/{len(viral_segments)}")
        print(f"Score: {segment.get('score', 0):.2f}")
        print(f"Duration: {segment.get('duration', 0):.1f}s")
        print(f"Text: {segment.get('highlight_text', '')[:80]}...")
        print(f"{'='*50}")
        
        output_path = os.path.join(output_dir, f"{base_name}_short_{i+1:02d}.mp4")
        
        # Create individual short
        success = create_vid_gear_short(
            highlights=[segment],
            input_video_path=input_video_path,
            output_video_path=output_path,
            **params
        )
        
        if success:
            created_videos.append(output_path)
            print(f"✅ Created: {output_path}")
        else:
            print(f"❌ Failed to create: {output_path}")
    
    print(f"\n🎉 Successfully created {len(created_videos)} short videos!")
    return created_videos


# Example usage
if __name__ == "__main__":
    # Example segments from your ViralShortsCreator
    example_segments = [
        {
            'start': 10.5,
            'end': 25.2,
            'duration': 14.7,
            'score': 8.5,
            'highlight_text': "This completely changed my life and I never expected this to happen",
            'segments': [
                {'start': 10.5, 'end': 15.0, 'text': "This completely changed my life"},
                {'start': 15.0, 'end': 20.0, 'text': "I never expected this to happen"},
                {'start': 20.0, 'end': 25.2, 'text': "Watch what happens next"}
            ]
        }
    ]
    
    # Custom parameters
    video_params = {
        'output_width': 1080,
        'output_height': 1920,
        'enable_face_tracking': True,
        'text_position': 'Bottom',
        'zoom_factor': 0.1,
        'crf_value': 23
    }
    
    # Create shorts
    created_videos = create_shorts_from_viral_segments(
        viral_segments=example_segments,
        input_video_path="input_video.mp4",
        output_dir="viral_shorts",
        **video_params
    )