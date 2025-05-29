import os
import yt_dlp
from ml_highlight_detect import create_viral_shorts_ml
from beast_gen import get_transcription

def prepare_media_files(url):
    """Download and prepare both video and audio for analysis"""
    
    # Create output directory
    os.makedirs('media', exist_ok=True)
    
    ydl_opts = {
        'format': 'best[height<=1080]',
        'outtmpl': 'media/input_video.%(ext)s',
        'merge_output_format': 'mp4',
        'postprocessors': [
            {
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }
        ],
        'keepvideo': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            
        # Rename files to standard names
        video_file = None
        audio_file = None
        
        for file in os.listdir('media'):
            if file.startswith('input_video.') and file.endswith('.mp4'):
                video_file = os.path.join('media', file)
            elif file.startswith('input_video.') and file.endswith('.mp3'):
                audio_file = os.path.join('media', file)
        
        # Standardize names
        if video_file and not video_file.endswith('input_video.mp4'):
            os.rename(video_file, 'media/input_video.mp4')
            video_file = 'media/input_video.mp4'
            
        if audio_file and not audio_file.endswith('input_audio.mp3'):
            os.rename(audio_file, 'media/input_audio.mp3')
            audio_file = 'media/input_audio.mp3'
        
        print(f"✅ Video: {video_file}")
        print(f"✅ Audio: {audio_file}")
        
        return {
            'video_path': video_file,
            'audio_path': audio_file,
            'info': info
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None



# Usage
if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=-4GmbBoYQjE&t=1s"
    media = prepare_media_files(url)
    if media:
        segments = get_transcription()
        
        # Now use with your viral detector
        shorts = create_viral_shorts_ml(
            whisper_segments=segments,
            audio_path=media['audio_path'],
            video_path=media['video_path']
        )