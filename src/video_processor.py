"""Video processing utilities using FFmpeg and OpenCV."""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
from rich.console import Console
from rich.progress import Progress, track

from .config import config

console = Console()

class VideoProcessor:
    """Handle video processing operations with FFmpeg and OpenCV."""
    
    def __init__(self, temp_dir: Optional[Path] = None):
        self.temp_dir = temp_dir or config.TEMP_DIR
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get video metadata using FFprobe."""
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', str(video_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            import json
            data = json.loads(result.stdout)
            
            video_stream = None
            audio_stream = None
            
            for stream in data['streams']:
                if stream['codec_type'] == 'video' and not video_stream:
                    video_stream = stream
                elif stream['codec_type'] == 'audio' and not audio_stream:
                    audio_stream = stream
            
            return {
                'duration': float(data['format'].get('duration', 0)),
                'size': int(data['format'].get('size', 0)),
                'bitrate': int(data['format'].get('bit_rate', 0)),
                'video': {
                    'codec': video_stream.get('codec_name', '') if video_stream else '',
                    'width': int(video_stream.get('width', 0)) if video_stream else 0,
                    'height': int(video_stream.get('height', 0)) if video_stream else 0,
                    'fps': eval(video_stream.get('r_frame_rate', '0/1')) if video_stream else 0,
                    'bitrate': int(video_stream.get('bit_rate', 0)) if video_stream else 0,
                } if video_stream else {},
                'audio': {
                    'codec': audio_stream.get('codec_name', '') if audio_stream else '',
                    'sample_rate': int(audio_stream.get('sample_rate', 0)) if audio_stream else 0,
                    'channels': int(audio_stream.get('channels', 0)) if audio_stream else 0,
                    'bitrate': int(audio_stream.get('bit_rate', 0)) if audio_stream else 0,
                } if audio_stream else {},
            }
            
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error getting video info: {e}[/red]")
            return {}
    
    def extract_audio(self, video_path: Path, output_path: Optional[Path] = None) -> Optional[Path]:
        """Extract audio from video file."""
        if not output_path:
            output_path = self.temp_dir / f"{video_path.stem}_audio.{config.AUDIO_FORMAT}"
        
        cmd = [
            'ffmpeg', '-i', str(video_path), '-vn', '-acodec', 'libmp3lame',
            '-ar', '16000', '-ac', '1', '-y', str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            console.print(f"[green]Audio extracted: {output_path}[/green]")
            return output_path
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Audio extraction failed: {e}[/red]")
            return None
    
    def extract_frames(self, video_path: Path, output_dir: Optional[Path] = None, 
                      fps: Optional[float] = None) -> List[Path]:
        """Extract frames from video at specified FPS."""
        if not output_dir:
            output_dir = self.temp_dir / f"{video_path.stem}_frames"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        fps = fps or config.FRAME_RATE
        
        cmd = [
            'ffmpeg', '-i', str(video_path), '-vf', f'fps={fps}',
            '-q:v', '2', '-y', str(output_dir / 'frame_%04d.jpg')
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            frame_files = sorted(output_dir.glob('frame_*.jpg'))
            console.print(f"[green]Extracted {len(frame_files)} frames[/green]")
            return frame_files
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Frame extraction failed: {e}[/red]")
            return []
    
    def create_clip(self, video_path: Path, start_time: float, duration: float,
                   output_path: Optional[Path] = None) -> Optional[Path]:
        """Create a clip from video between start_time and start_time + duration."""
        if not output_path:
            output_path = self.temp_dir / f"{video_path.stem}_clip_{start_time:.1f}s.{config.VIDEO_FORMAT}"
        
        cmd = [
            'ffmpeg', '-i', str(video_path), '-ss', str(start_time),
            '-t', str(duration), '-c:v', 'libx264', '-c:a', 'aac',
            '-preset', 'fast', '-crf', '23', '-y', str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            console.print(f"[green]Clip created: {output_path}[/green]")
            return output_path
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Clip creation failed: {e}[/red]")
            return None
    
    def add_subtitles(self, video_path: Path, subtitle_path: Path,
                     output_path: Optional[Path] = None) -> Optional[Path]:
        """Add subtitles to video."""
        if not output_path:
            output_path = self.temp_dir / f"{video_path.stem}_subtitled.{config.VIDEO_FORMAT}"
        
        cmd = [
            'ffmpeg', '-i', str(video_path), '-vf', f"subtitles='{subtitle_path}'",
            '-c:a', 'copy', '-y', str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            console.print(f"[green]Subtitles added: {output_path}[/green]")
            return output_path
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Subtitle addition failed: {e}[/red]")
            return None
    
    def resize_video(self, video_path: Path, width: int, height: int,
                    output_path: Optional[Path] = None) -> Optional[Path]:
        """Resize video to specified dimensions."""
        if not output_path:
            output_path = self.temp_dir / f"{video_path.stem}_{width}x{height}.{config.VIDEO_FORMAT}"
        
        cmd = [
            'ffmpeg', '-i', str(video_path), '-vf', f'scale={width}:{height}',
            '-c:a', 'copy', '-y', str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            console.print(f"[green]Video resized: {output_path}[/green]")
            return output_path
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Video resize failed: {e}[/red]")
            return None
    
    def add_fade_effects(self, video_path: Path, fade_in: float = 1.0, fade_out: float = 1.0,
                        output_path: Optional[Path] = None) -> Optional[Path]:
        """Add fade in/out effects to video."""
        if not output_path:
            output_path = self.temp_dir / f"{video_path.stem}_faded.{config.VIDEO_FORMAT}"
        
        info = self.get_video_info(video_path)
        duration = info.get('duration', 0)
        
        if duration <= 0:
            console.print("[red]Cannot add fade effects: invalid duration[/red]")
            return None
        
        fade_out_start = duration - fade_out
        
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vf', f'fade=t=in:st=0:d={fade_in},fade=t=out:st={fade_out_start}:d={fade_out}',
            '-af', f'afade=t=in:st=0:d={fade_in},afade=t=out:st={fade_out_start}:d={fade_out}',
            '-y', str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            console.print(f"[green]Fade effects added: {output_path}[/green]")
            return output_path
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Fade effects failed: {e}[/red]")
            return None
    
    def enhance_audio(self, video_path: Path, output_path: Optional[Path] = None) -> Optional[Path]:
        """Enhance audio with normalization and noise reduction."""
        if not output_path:
            output_path = self.temp_dir / f"{video_path.stem}_enhanced.{config.VIDEO_FORMAT}"
        
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11,highpass=f=200,lowpass=f=3000',
            '-c:v', 'copy', '-y', str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            console.print(f"[green]Audio enhanced: {output_path}[/green]")
            return output_path
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Audio enhancement failed: {e}[/red]")
            return None
    
    def create_thumbnail(self, video_path: Path, timestamp: float = 0.0,
                        output_path: Optional[Path] = None) -> Optional[Path]:
        """Create thumbnail from video at specified timestamp."""
        if not output_path:
            output_path = self.temp_dir / f"{video_path.stem}_thumb.jpg"
        
        cmd = [
            'ffmpeg', '-i', str(video_path), '-ss', str(timestamp),
            '-vframes', '1', '-q:v', '2', '-y', str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            console.print(f"[green]Thumbnail created: {output_path}[/green]")
            return output_path
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Thumbnail creation failed: {e}[/red]")
            return None
    
    def get_frame_at_time(self, video_path: Path, timestamp: float) -> Optional[np.ndarray]:
        """Get a specific frame at timestamp using OpenCV."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            console.print(f"[red]Cannot open video: {video_path}[/red]")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            console.print(f"[red]Cannot read frame at {timestamp}s[/red]")
            return None
    
    def cleanup_temp_files(self):
        """Clean up temporary processing files."""
        for file in self.temp_dir.glob("*"):
            if file.is_file():
                try:
                    file.unlink()
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not delete {file.name}: {e}[/yellow]")