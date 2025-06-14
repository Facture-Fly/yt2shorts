import os
import numpy as np
from collections import deque
from tqdm import tqdm
from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip, ColorClip
from moviepy.video.fx import resize, crop
import cv2




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
        
    output_path = os.path.join(output_dir, f"{base_name}_short_generated.mp4")
        
        # Create individual short
    success = create_moviepy_short(
        highlights=viral_segments,
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

def create_moviepy_short(highlights, input_video_path, output_video_path=None, **params):
    """
    Create short video with MoviePy from extracted segments
    
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
    TEXT_BG_OPACITY = params.get('text_bg_opacity', 0.8)
    MAX_SEGMENT_DURATION = params.get('max_segment_duration', 15)  # seconds
    
    print(f"Creating shorts from {len(highlights)} segments...")
    print(f"Output: {output_video_path}")
    
    # Initialize face detection
    face_cascade = None
    if ENABLE_FACE_TRACKING:
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
    
    try:
        # Load the video
        video = VideoFileClip(input_video_path)
        video_duration = video.duration
        fps = video.fps
        
        print(f"Video info: {fps:.1f} FPS, {video_duration:.1f}s duration")
        
        # Process each highlight segment
        video_clips = []
        
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
            
            # Extract the segment
            segment_clip = video.subclip(start_time, start_time + duration)
            
            # Apply transformations
            processed_clip = apply_moviepy_effects(
                segment_clip, 
                TARGET_WIDTH, 
                TARGET_HEIGHT, 
                ZOOM_FACTOR, 
                ZOOM_DURATION,
                ENABLE_FACE_TRACKING,
                face_cascade,
                highlight
            )
            
            # Add text overlay if configured
            if TEXT_LINES or highlight.get('highlight_text'):
                processed_clip = add_moviepy_text(
                    processed_clip, 
                    highlight, 
                    TEXT_LINES, 
                    TEXT_POSITION, 
                    TEXT_BG_OPACITY,
                    TARGET_WIDTH,
                    TARGET_HEIGHT
                )
            
            video_clips.append(processed_clip)
        
        if not video_clips:
            print("No valid segments to process")
            video.close()
            return False
        
        # Concatenate all clips
        print("\nConcatenating segments...")
        final_clip = CompositeVideoClip(video_clips).set_duration(sum(clip.duration for clip in video_clips))
        
        # Ensure target dimensions
        final_clip = final_clip.resize((TARGET_WIDTH, TARGET_HEIGHT))
        
        # Write the final video
        print(f"Writing output video: {output_video_path}")
        final_clip.write_videofile(
            output_video_path,
            codec='libx264',
            audio_codec='aac',
            fps=fps,
            preset='medium',
            ffmpeg_params=['-crf', '23', '-pix_fmt', 'yuv420p']
        )
        
        # Cleanup
        final_clip.close()
        video.close()
        
        print(f"\nShort video created: {output_video_path}")
        return True
        
    except Exception as e:
        print(f"Error during processing: {e}")
        return False


def apply_moviepy_effects(clip, target_width, target_height, zoom_factor, zoom_duration, enable_face_tracking, face_cascade, segment_info):
    """Apply visual effects using MoviePy"""
    try:
        # Get clip dimensions
        w, h = clip.size
        duration = clip.duration
        
        # Calculate aspect ratio preserving resize
        aspect_ratio = w / h
        target_aspect = target_width / target_height
        
        if aspect_ratio > target_aspect:
            # Video is wider - fit to height
            new_height = target_height
            new_width = int(new_height * aspect_ratio)
        else:
            # Video is taller - fit to width
            new_width = target_width
            new_height = int(new_width / aspect_ratio)
        
        # Resize to calculated dimensions
        clip = clip.resize((new_width, new_height))
        
        # Apply zoom effect at start and end
        def zoom_effect(get_frame, t):
            frame = get_frame(t)
            
            # Calculate zoom based on time
            if t < zoom_duration or t > duration - zoom_duration:
                if t < zoom_duration:
                    progress = t / zoom_duration
                else:
                    progress = (duration - t) / zoom_duration
                
                zoom = 1 + zoom_factor * (1 - np.cos(progress * np.pi)) / 2
                
                # Apply zoom by resizing
                zoomed_height = int(frame.shape[0] * zoom)
                zoomed_width = int(frame.shape[1] * zoom)
                
                if zoomed_width > 0 and zoomed_height > 0:
                    import cv2
                    frame = cv2.resize(frame, (zoomed_width, zoomed_height))
                    
                    # Center crop after zoom
                    fh, fw = frame.shape[:2]
                    start_y = max(0, fh//2 - target_height//2)
                    end_y = min(fh, start_y + target_height)
                    start_x = max(0, fw//2 - target_width//2)
                    end_x = min(fw, start_x + target_width)
                    
                    frame = frame[start_y:end_y, start_x:end_x]
            
            return frame
        
        clip = clip.fl(zoom_effect)
        
        # Apply face tracking crop if enabled and needed
        if enable_face_tracking and face_cascade and new_width > target_width:
            clip = apply_face_tracking_crop(clip, target_width, target_height, face_cascade)
        else:
            # Center crop if needed
            if new_width > target_width or new_height > target_height:
                clip = crop(clip, 
                           x_center=new_width//2, 
                           y_center=new_height//2,
                           width=min(target_width, new_width),
                           height=min(target_height, new_height))
        
        return clip
        
    except Exception as e:
        print(f"Error applying effects: {e}")
        return clip.resize((target_width, target_height))


def apply_face_tracking_crop(clip, target_width, target_height, face_cascade):
    """Apply dynamic face tracking crop using MoviePy"""
    position_buffer = deque(maxlen=5)
    prev_x = target_width // 2
    
    def face_crop_effect(get_frame, t):
        nonlocal prev_x
        frame = get_frame(t)
        
        try:
            h, w = frame.shape[:2]
            
            if w <= target_width:
                return frame
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
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
            max_delta = 400 * (1/30)  # Assuming 30fps for smoothing
            smoothed_x = np.clip(smoothed_x, prev_x - max_delta, prev_x + max_delta)
            prev_x = smoothed_x
            
            # Calculate crop bounds
            left = int(smoothed_x - target_width//2)
            left = max(0, min(left, w - target_width))
            
            return frame[:, left:left + target_width]
            
        except Exception as e:
            print(f"Error in face tracking: {e}")
            # Fallback to center crop
            start_x = (frame.shape[1] - target_width) // 2
            return frame[:, start_x:start_x + target_width]
    
    return clip.fl(face_crop_effect)


def add_moviepy_text(clip, segment_info, text_lines, text_position, text_bg_opacity, target_width, target_height):
    """Add text overlay using MoviePy - with ImageMagick fallback"""
    try:
        # Prepare text lines
        final_text_lines = text_lines.copy() if text_lines else []
        
        # Add segment text if available and no custom text provided
        if not final_text_lines and segment_info and segment_info.get('highlight_text'):
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
                final_text_lines.append({
                    'text': line,
                    'fontsize': 50,
                    'color': 'white'
                })
        
        if not final_text_lines:
            return clip
        
        # Create text clips
        text_clips = []
        
        # Calculate position based on TEXT_POSITION
        if text_position == "Top":
            y_start = 100
        elif text_position == "Center":
            y_start = (target_height - len(final_text_lines) * 60) // 2
        else:  # Bottom
            y_start = target_height - len(final_text_lines) * 60 - 100
        
        # Create background
        if text_bg_opacity > 0:
            bg_height = len(final_text_lines) * 60 + 40
            bg_clip = ColorClip(
                size=(target_width - 40, bg_height),
                color=(0, 0, 0)
            ).set_opacity(text_bg_opacity).set_duration(clip.duration).set_position(('center', y_start - 20))
            text_clips.append(bg_clip)
        
        # Create text clips with fallback for ImageMagick issues
        for i, text_config in enumerate(final_text_lines):
            if not text_config.get('text'):
                continue
            
            try:
                # Try with stroke first
                text_clip = TextClip(
                    text_config['text'],
                    fontsize=text_config.get('fontsize', 50),
                    color=text_config.get('color', 'white'),
                    font='Arial-Bold',
                    stroke_color='black',
                    stroke_width=2
                ).set_duration(clip.duration).set_position(('center', y_start + i * 60))
            except Exception as text_error:
                print(f"Warning: TextClip with stroke failed: {text_error}")
                try:
                    # Fallback without stroke
                    text_clip = TextClip(
                        text_config['text'],
                        fontsize=text_config.get('fontsize', 50),
                        color=text_config.get('color', 'white'),
                        font='Arial-Bold'
                    ).set_duration(clip.duration).set_position(('center', y_start + i * 60))
                except Exception as text_error2:
                    print(f"Warning: TextClip fallback failed: {text_error2}")
                    # Skip text if all attempts fail
                    continue
            
            text_clips.append(text_clip)
        
        # Composite with text
        if text_clips:
            return CompositeVideoClip([clip] + text_clips)
        else:
            return clip
        
    except Exception as e:
        print(f"Error adding text: {e}")
        return clip


def extend_segment_to_target_duration(segment, target_duration, input_video_path):
    """Extend a short segment to target duration by adding surrounding content"""
    try:
        # Get video duration to avoid going beyond bounds
        video = VideoFileClip(input_video_path)
        video_duration = video.duration
        video.close()
        
        # Current segment info
        original_start = segment.get('start', 0)
        original_end = segment.get('end', original_start + 5)
        original_duration = original_end - original_start
        
        # If already longer than target, return as-is
        if original_duration >= target_duration:
            return segment
        
        # Calculate how much to extend
        additional_time = target_duration - original_duration
        
        # Try to extend equally before and after
        extension_before = additional_time / 2
        extension_after = additional_time / 2
        
        # Calculate new bounds
        new_start = max(0, original_start - extension_before)
        new_end = min(video_duration, original_end + extension_after)
        
        # If we hit a boundary, redistribute the extension
        if new_start == 0:
            # Hit start boundary, extend more at the end
            new_end = min(video_duration, original_end + additional_time)
        elif new_end == video_duration:
            # Hit end boundary, extend more at the beginning
            new_start = max(0, original_start - additional_time)
        
        # Create extended segment
        extended_segment = segment.copy()
        extended_segment['start'] = new_start
        extended_segment['end'] = new_end
        extended_segment['duration'] = new_end - new_start
        
        print(f"  Extended from {original_duration:.1f}s to {extended_segment['duration']:.1f}s")
        print(f"  New time range: {new_start:.1f}s - {new_end:.1f}s")
        
        return extended_segment
        
    except Exception as e:
        print(f"Warning: Could not extend segment: {e}")
        return segment


def create_single_clip_from_segment(segment, input_video_path, **params):
    """Create a single processed clip from a segment"""
    try:
        # Extract parameters
        TARGET_WIDTH = params.get('output_width', 1080)
        TARGET_HEIGHT = params.get('output_height', 1920)
        ZOOM_FACTOR = params.get('zoom_factor', 0.08)
        ZOOM_DURATION = params.get('zoom_duration', 15)
        ENABLE_FACE_TRACKING = params.get('enable_face_tracking', True)
        TEXT_LINES = params.get('text_lines', [])
        TEXT_POSITION = params.get('text_position', 'Top')
        TEXT_BG_OPACITY = params.get('text_bg_opacity', 0.8)
        
        # Load video and extract segment
        video = VideoFileClip(input_video_path)
        start_time = segment.get('start', 0)
        end_time = segment.get('end', start_time + 60)
        
        # Extract the segment clip
        segment_clip = video.subclip(start_time, end_time)
        video.close()
        
        # Initialize face detection if needed
        face_cascade = None
        if ENABLE_FACE_TRACKING:
            try:
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                if face_cascade.empty():
                    ENABLE_FACE_TRACKING = False
            except Exception:
                ENABLE_FACE_TRACKING = False
        
        # Apply transformations
        processed_clip = apply_moviepy_effects(
            segment_clip, 
            TARGET_WIDTH, 
            TARGET_HEIGHT, 
            ZOOM_FACTOR, 
            ZOOM_DURATION,
            ENABLE_FACE_TRACKING,
            face_cascade,
            segment
        )
        
        # Add text overlay if configured
        if TEXT_LINES or segment.get('highlight_text'):
            processed_clip = add_moviepy_text(
                processed_clip, 
                segment, 
                TEXT_LINES, 
                TEXT_POSITION, 
                TEXT_BG_OPACITY,
                TARGET_WIDTH,
                TARGET_HEIGHT
            )
        
        # Ensure target dimensions
        processed_clip = processed_clip.resize((TARGET_WIDTH, TARGET_HEIGHT))
        
        return processed_clip
        
    except Exception as e:
        print(f"Error creating clip from segment: {e}")
        return None


def create_shorts_from_viral_segments(viral_segments, input_video_path, output_dir="shorts", **params):
    """
    Create a single concatenated video from all viral segments using MoviePy
    
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
    
    base_name = os.path.splitext(os.path.basename(input_video_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_compiled_shorts.mp4")
    
    print(f"\n{'='*60}")
    print(f"Creating compiled short from {len(viral_segments)} segments")
    print(f"Output: {output_path}")
    print(f"{'='*60}")
    
    # Process all segments and collect clips
    processed_clips = []
    total_target_duration = 0
    
    for i, segment in enumerate(viral_segments):
        print(f"\n📹 Processing Segment {i+1}/{len(viral_segments)}")
        print(f"   Score: {segment.get('score', 0):.2f}")
        print(f"   Original Duration: {segment.get('duration', 0):.1f}s")
        print(f"   Text: {segment.get('highlight_text', '')[:60]}...")
        
        # Extend segment to target duration per segment (default 60 seconds)
        segment_target_duration = params.get('segment_duration', 60)
        extended_segment = extend_segment_to_target_duration(
            segment, segment_target_duration, input_video_path
        )
        
        total_target_duration += extended_segment['duration']
        
        # Create clip for this segment
        clip = create_single_clip_from_segment(
            extended_segment, 
            input_video_path, 
            **params
        )
        
        if clip:
            processed_clips.append(clip)
            print(f"   ✅ Processed segment {i+1}")
        else:
            print(f"   ❌ Failed to process segment {i+1}")
    
    if not processed_clips:
        print("❌ No clips were successfully processed")
        return []
    
    print(f"\n🔗 Concatenating {len(processed_clips)} clips...")
    print(f"📏 Total duration: {total_target_duration:.1f}s ({total_target_duration/60:.1f} minutes)")
    
    try:
        # Concatenate all clips
        from moviepy.editor import concatenate_videoclips
        final_clip = concatenate_videoclips(processed_clips, method="compose")
        
        # Get video info for output settings
        video = VideoFileClip(input_video_path)
        fps = video.fps
        video.close()
        
        # Write the final compiled video
        print(f"\n🎬 Writing final compiled video...")
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            fps=fps,
            preset='medium',
            ffmpeg_params=['-crf', '23', '-pix_fmt', 'yuv420p']
        )
        
        # Cleanup
        for clip in processed_clips:
            clip.close()
        final_clip.close()
        
        print(f"\n🎉 Successfully created compiled short: {output_path}")
        return [output_path]
        
    except Exception as e:
        print(f"❌ Error creating compiled video: {e}")
        # Cleanup on error
        for clip in processed_clips:
            try:
                clip.close()
            except:
                pass
        return []


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
        'max_segment_duration': 15
    }
    
    # Create shorts
    created_videos = create_shorts_from_viral_segments(
        viral_segments=example_segments,
        input_video_path="input_video.mp4",
        output_dir="viral_shorts",
        **video_params
    )