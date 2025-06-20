"""Video downloader using yt-dlp."""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urlparse

import yt_dlp
from rich.console import Console
from rich.progress import Progress, TaskID

from .config import config

console = Console()

class VideoDownloader:
    """Download videos from YouTube and other platforms using yt-dlp."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or config.TEMP_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def get_video_info(self, url: str) -> Dict[str, Any]:
        """Get video metadata without downloading."""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', 'Unknown'),
                    'view_count': info.get('view_count', 0),
                    'like_count': info.get('like_count', 0),
                    'upload_date': info.get('upload_date', ''),
                    'description': info.get('description', ''),
                    'thumbnail': info.get('thumbnail', ''),
                    'id': info.get('id', ''),
                }
            except Exception as e:
                console.print(f"[red]Error extracting video info: {e}[/red]")
                return {}
    
    def download_video(self, url: str, filename: Optional[str] = None) -> Optional[Path]:
        """Download video in best quality."""
        if not filename:
            # Generate filename from video ID
            info = self.get_video_info(url)
            video_id = info.get('id', 'video')
            filename = f"{video_id}.{config.VIDEO_FORMAT}"
        
        output_path = self.output_dir / filename
        
        ydl_opts = {
            'format': f'bestvideo[ext={config.VIDEO_FORMAT}]+bestaudio[ext=m4a]/best[ext={config.VIDEO_FORMAT}]/best',
            'outtmpl': str(output_path),
            'merge_output_format': config.VIDEO_FORMAT,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'ignoreerrors': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                console.print(f"[blue]Downloading video from: {url}[/blue]")
                ydl.download([url])
                
                if output_path.exists():
                    console.print(f"[green]Video downloaded: {output_path}[/green]")
                    return output_path
                else:
                    console.print("[red]Download failed - file not found[/red]")
                    return None
                    
            except Exception as e:
                console.print(f"[red]Download error: {e}[/red]")
                return None
    
    def download_audio_only(self, url: str, filename: Optional[str] = None) -> Optional[Path]:
        """Download audio stream only."""
        if not filename:
            info = self.get_video_info(url)
            video_id = info.get('id', 'audio')
            filename = f"{video_id}.{config.AUDIO_FORMAT}"
        
        output_path = self.output_dir / filename
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(output_path),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': config.AUDIO_FORMAT,
                'preferredquality': '192',
            }],
            'ignoreerrors': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                console.print(f"[blue]Downloading audio from: {url}[/blue]")
                ydl.download([url])
                
                if output_path.exists():
                    console.print(f"[green]Audio downloaded: {output_path}[/green]")
                    return output_path
                else:
                    console.print("[red]Audio download failed - file not found[/red]")
                    return None
                    
            except Exception as e:
                console.print(f"[red]Audio download error: {e}[/red]")
                return None
    
    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and supported."""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Check with yt-dlp
            ydl_opts = {'quiet': True, 'no_warnings': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(url, download=False)
                return True
                
        except Exception:
            return False
    
    def cleanup_temp_files(self):
        """Clean up temporary download files."""
        for file in self.output_dir.glob("*"):
            if file.is_file():
                try:
                    file.unlink()
                    console.print(f"[dim]Cleaned up: {file.name}[/dim]")
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not delete {file.name}: {e}[/yellow]")

def download_video(url: str, output_dir: Optional[Path] = None) -> Optional[Path]:
    """Convenience function to download a video."""
    downloader = VideoDownloader(output_dir)
    return downloader.download_video(url)

def get_video_info(url: str) -> Dict[str, Any]:
    """Convenience function to get video information."""
    downloader = VideoDownloader()
    return downloader.get_video_info(url)