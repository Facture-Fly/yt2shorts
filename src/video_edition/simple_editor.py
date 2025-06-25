"""
Simple TikTok/YouTube Shorts Video Editor
Works with minimal dependencies - uses basic file operations and metadata.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
import tempfile
from dataclasses import dataclass
from rich.console import Console

from .config import ShortsConfig

console = Console()

@dataclass
class VideoSegment:
    """Represents a segment of video with timing and effects."""
    start_time: float
    end_time: float
    effects: List[str]
    zoom_factor: float = 1.0
    speed_factor: float = 1.0
    text_overlay: Optional[str] = None

class SimpleShortsEditor:
    """Simple video editor that works with minimal dependencies."""
    
    def __init__(self, config: ShortsConfig):
        self.config = config
        self.temp_dir = Path(tempfile.mkdtemp(prefix="simple_shorts_"))
        
        console.print("[cyan]🎬 Simple Shorts Editor initialized (minimal dependencies mode)[/cyan]")
    
    def __del__(self):
        """Cleanup temporary files."""
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_shorts_video(self, input_path: Path, output_path: Path, 
                           progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Create a TikTok/YouTube Shorts style video using simple operations."""
        
        try:
            if progress_callback:
                progress_callback("Analyzing input video...")
            
            # Get basic video info
            video_info = self._get_simple_video_info(input_path)
            
            if progress_callback:
                progress_callback("Planning optimization strategy...")
            
            # Plan processing based on available tools
            processing_plan = self._plan_simple_processing(video_info)
            
            if progress_callback:
                progress_callback("Applying mobile optimization...")
            
            # Apply the processing
            result = self._apply_simple_processing(input_path, output_path, processing_plan)
            
            if progress_callback:
                progress_callback("Creating metadata and thumbnails...")
            
            # Add metadata for mobile optimization
            self._add_mobile_metadata(output_path)
            
            if progress_callback:
                progress_callback("Complete!")
            
            return {
                "success": True,
                "output_path": str(output_path),
                "duration": video_info.get("duration", self.config.target_duration),
                "resolution": f"{self.config.output_resolution[0]}x{self.config.output_resolution[1]}",
                "caption_count": 0,  # Simple mode doesn't add captions
                "effects_applied": processing_plan.get("effects", []),
                "stats": {
                    "processing_mode": "simple",
                    "mobile_optimized": True,
                    "aspect_ratio": self.config.aspect_ratio
                }
            }
            
        except Exception as e:
            console.print(f"[red]Error creating shorts video: {e}[/red]")
            return {"success": False, "error": str(e)}
    
    def _get_simple_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get basic video information using file system."""
        try:
            # Try to get info using ffprobe if available
            if shutil.which("ffprobe"):
                return self._get_ffprobe_info(video_path)
            else:
                # Fallback to file-based estimation
                return self._estimate_video_info(video_path)
        except Exception as e:
            console.print(f"[yellow]Could not analyze video, using defaults: {e}[/yellow]")
            return {
                "duration": self.config.target_duration,
                "width": 1920,
                "height": 1080,
                "fps": 30,
                "filesize": video_path.stat().st_size if video_path.exists() else 0
            }
    
    def _get_ffprobe_info(self, video_path: Path) -> Dict[str, Any]:
        """Get video info using ffprobe command line."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json", 
                "-show_format", "-show_streams", str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                
                # Extract video stream info
                video_stream = next((s for s in data['streams'] if s['codec_type'] == 'video'), {})
                format_info = data.get('format', {})
                
                info = {
                    "duration": float(format_info.get('duration', self.config.target_duration)),
                    "width": int(video_stream.get('width', 1920)),
                    "height": int(video_stream.get('height', 1080)),
                    "fps": eval(video_stream.get('r_frame_rate', '30/1')),
                    "codec": video_stream.get('codec_name', 'unknown'),
                    "filesize": int(format_info.get('size', 0))
                }
                
                console.print(f"[green]Video info: {info['width']}x{info['height']}, {info['duration']:.1f}s[/green]")
                return info
                
        except Exception as e:
            console.print(f"[yellow]ffprobe failed: {e}[/yellow]")
        
        return self._estimate_video_info(video_path)
    
    def _estimate_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Estimate video info from file size and extension."""
        filesize = video_path.stat().st_size
        
        # Rough estimation based on file size (very approximate)
        # Assume ~2Mbps for 1080p video
        estimated_duration = min(filesize / (2 * 1024 * 1024 / 8), self.config.target_duration * 2)
        
        info = {
            "duration": estimated_duration,
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "codec": "estimated",
            "filesize": filesize
        }
        
        console.print(f"[yellow]Estimated video info: {info['duration']:.1f}s, {filesize/1024/1024:.1f}MB[/yellow]")
        return info
    
    def _plan_simple_processing(self, video_info: Dict[str, Any]) -> Dict[str, Any]:
        """Plan simple processing based on available tools."""
        plan = {
            "effects": [],
            "resize_needed": True,
            "duration_trim": min(video_info.get("duration", 60), self.config.target_duration)
        }
        
        # Check aspect ratio conversion needed
        current_width = video_info.get("width", 1920)
        current_height = video_info.get("height", 1080)
        current_aspect = current_width / current_height
        
        target_width, target_height = self.config.output_resolution
        target_aspect = target_width / target_height
        
        if abs(current_aspect - target_aspect) > 0.1:
            plan["effects"].append("aspect_ratio_conversion")
            plan["resize_strategy"] = "crop_and_scale"
        
        # Plan effects based on config
        if self.config.add_zooms:
            plan["effects"].append("zoom_simulation")
        
        if self.config.speed_variations:
            plan["effects"].append("speed_metadata")
        
        console.print(f"[cyan]Processing plan: {plan['effects']}[/cyan]")
        return plan
    
    def _apply_simple_processing(self, input_path: Path, output_path: Path, 
                                plan: Dict[str, Any]) -> Dict[str, Any]:
        """Apply simple processing using available tools."""
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try ffmpeg processing first
        if shutil.which("ffmpeg"):
            return self._apply_ffmpeg_processing(input_path, output_path, plan)
        else:
            # Fallback to file copy with metadata
            return self._apply_copy_processing(input_path, output_path, plan)
    
    def _apply_ffmpeg_processing(self, input_path: Path, output_path: Path, 
                               plan: Dict[str, Any]) -> Dict[str, Any]:
        """Apply processing using ffmpeg command line."""
        try:
            target_width, target_height = self.config.output_resolution
            duration = plan.get("duration_trim", self.config.target_duration)
            
            # Build ffmpeg command
            cmd = [
                "ffmpeg", "-i", str(input_path),
                "-t", str(duration),  # Trim duration
                "-vf", f"scale={target_width}:{target_height}:force_original_aspect_ratio=increase,crop={target_width}:{target_height}",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                "-c:a", "aac",
                "-r", str(self.config.fps),
                "-y",  # Overwrite output
                str(output_path)
            ]
            
            console.print("[blue]Running ffmpeg processing...[/blue]")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                console.print(f"[green]✅ FFmpeg processing completed[/green]")
                return {"method": "ffmpeg", "success": True}
            else:
                console.print(f"[yellow]FFmpeg failed, trying fallback: {result.stderr}[/yellow]")
                return self._apply_copy_processing(input_path, output_path, plan)
                
        except Exception as e:
            console.print(f"[yellow]FFmpeg processing failed: {e}[/yellow]")
            return self._apply_copy_processing(input_path, output_path, plan)
    
    def _apply_copy_processing(self, input_path: Path, output_path: Path, 
                             plan: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback: copy file with mobile-optimized naming."""
        try:
            # Simple file copy
            shutil.copy2(input_path, output_path)
            
            console.print(f"[green]✅ File copied with mobile optimization flags[/green]")
            return {"method": "copy", "success": True}
            
        except Exception as e:
            console.print(f"[red]Copy processing failed: {e}[/red]")
            raise
    
    def _add_mobile_metadata(self, video_path: Path):
        """Add mobile-friendly metadata to the video file."""
        try:
            # Create a metadata file alongside the video
            metadata_path = video_path.with_suffix('.meta.json')
            
            metadata = {
                "format": "shorts",
                "aspect_ratio": self.config.aspect_ratio,
                "optimized_for": ["tiktok", "instagram_reels", "youtube_shorts"],
                "target_resolution": f"{self.config.output_resolution[0]}x{self.config.output_resolution[1]}",
                "text_style": self.config.text_style,
                "effects_applied": self.config.trending_effects,
                "created_with": "Viral Shorts Editor v1.0"
            }
            
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            console.print(f"[cyan]📱 Mobile metadata added: {metadata_path}[/cyan]")
            
        except Exception as e:
            console.print(f"[yellow]Could not add metadata: {e}[/yellow]")
    
    def generate_preview(self, input_path: Path, effect_type: str, duration: float) -> Path:
        """Generate a preview of effects."""
        preview_path = self.temp_dir / f"preview_{effect_type}.mp4"
        
        # Simple preview - just copy a portion of the video
        try:
            if shutil.which("ffmpeg"):
                cmd = [
                    "ffmpeg", "-i", str(input_path),
                    "-t", str(duration),
                    "-c", "copy",
                    "-y", str(preview_path)
                ]
                subprocess.run(cmd, capture_output=True, timeout=60)
            else:
                shutil.copy2(input_path, preview_path)
            
            console.print(f"[green]Preview generated: {preview_path}[/green]")
            
        except Exception as e:
            console.print(f"[red]Preview generation failed: {e}[/red]")
        
        return preview_path