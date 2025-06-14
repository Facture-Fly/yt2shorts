import os
import yt_dlp
#from ml_highlight_detect import create_viral_shorts_ml
from safe_highlight_detect import create_viral_shorts_ml
from transcription import get_transcription
from create_short_v2 import create_shorts_from_viral_segments

def prepare_media_files(url):
    """Download and prepare both video and audio for analysis"""
    
    # Create output directory
    os.makedirs('media', exist_ok=True)
    
    ydl_opts = {
        'format': 'best[height<=1080]',
        'outtmpl': 'media/input_video.%(ext)s',
        'merge_output_format': 'mp4',
        'keepvideo': True,
        # Remove audio postprocessor - we'll extract manually
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            
        # Find video file
        video_file = None
        for file in os.listdir('media'):
            if file.startswith('input_video.') and file.endswith('.mp4'):
                video_file = os.path.join('media', file)
                break
        
        if not video_file:
            raise Exception("Video file not found")
        
        # Standardize video name
        if not video_file.endswith('input_video.mp4'):
            os.rename(video_file, 'media/input_video.mp4')
            video_file = 'media/input_video.mp4'
        
        # Extract audio manually using ffmpeg (more reliable)
        audio_file = 'media/input_audio.mp3'
        import subprocess
        
        print("🔄 Extracting audio with ffmpeg...")
        cmd = [
            'ffmpeg', '-i', video_file,
            '-q:a', '0',        # Best quality
            '-map', 'a',        # Map audio stream
            '-y',               # Overwrite
            audio_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"⚠️ FFmpeg audio extraction failed: {result.stderr}")
            # Fallback: try with different settings
            cmd = [
                'ffmpeg', '-i', video_file,
                '-acodec', 'libmp3lame',
                '-b:a', '192k',
                '-y',
                audio_file
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Audio extracted successfully")
        else:
            print(f"❌ Audio extraction failed")
            audio_file = None
        
        # Verify durations match
        if audio_file and os.path.exists(audio_file):
            video_duration = get_duration(video_file)
            audio_duration = get_duration(audio_file)
            
            print(f"🎬 Video: {video_duration:.1f}s")
            print(f"🎵 Audio: {audio_duration:.1f}s")
            
            if abs(video_duration - audio_duration) > 1:
                print(f"⚠️ Duration mismatch: {abs(video_duration - audio_duration):.1f}s difference")
            else:
                print(f"✅ Durations match!")
        
        return {
            'video_path': video_file,
            'audio_path': audio_file,
            'info': info
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def get_duration(file_path):
    """Get media file duration"""
    import subprocess
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-show_entries', 
            'format=duration', '-of', 'csv=p=0', file_path
        ], capture_output=True, text=True)
        return float(result.stdout.strip())
    except:
        return 0



# Usage
if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=-4GmbBoYQjE&t=1s"
    media = prepare_media_files(url)
    if not media:
        raise ValueError("No media to analyse")
    
    segments = get_transcription(input_file=media['audio_path'])
    
    # Now use with your viral detector
    shorts = create_viral_shorts_ml(
        whisper_segments=segments,
        audio_path=media['audio_path'],
        video_path=media['video_path']
    )
    
    video_params = {
        'output_width': 1080,
        'output_height': 1920,
        'enable_face_tracking': True,
        'text_position': 'Bottom',
        'zoom_factor': 0.1,
        'crf_value': 23,
        'segment_duration': 60  # Each segment will be 60 seconds
    }

    # Create shorts
    created_videos = create_shorts_from_viral_segments(
        viral_segments=shorts,
        input_video_path=media['video_path'],
        output_dir="viral_shorts",
        **video_params
    )