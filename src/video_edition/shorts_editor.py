"""
TikTok/YouTube Shorts Video Editor
Creates mobile-optimized viral videos with captions, effects, and trending elements.
"""

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Create a minimal numpy substitute for basic operations
    class np:
        @staticmethod
        def arange(start, stop, step=1):
            result = []
            current = start
            while current < stop:
                result.append(current)
                current += step
            return result
        
        @staticmethod
        def mean(arr):
            return sum(arr) / len(arr) if arr else 0

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
import subprocess
import json
import tempfile
import shutil
from dataclasses import dataclass
from rich.console import Console
from rich.progress import Progress

from .config import ShortsConfig, TRENDING_EFFECTS

# Try to import analysis tools
try:
    from ..speech_to_text import SpeechToText, TranscriptSegment
    SPEECH_TO_TEXT_AVAILABLE = True
except ImportError:
    SPEECH_TO_TEXT_AVAILABLE = False
    TranscriptSegment = None

try:
    from ..visual_analysis import VisualAnalyzer
    VISUAL_ANALYSIS_AVAILABLE = True
except ImportError:
    VISUAL_ANALYSIS_AVAILABLE = False

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

@dataclass
class CaptionSegment:
    """Represents a caption with styling and timing."""
    text: str
    start_time: float
    end_time: float
    style: Dict[str, Any]
    position: Tuple[int, int]

class ShortsVideoEditor:
    """Main video editor for creating TikTok/YouTube Shorts style content."""
    
    def __init__(self, config: ShortsConfig):
        self.config = config
        self.temp_dir = Path(tempfile.mkdtemp(prefix="shorts_editor_"))
        self.speech_to_text = None
        self.visual_analyzer = None
        
        # Check dependencies
        if not FFMPEG_AVAILABLE:
            console.print("[yellow]Warning: ffmpeg-python not available. Video processing may be limited.[/yellow]")
        
        # Initialize analysis tools if needed and available
        if (config.add_captions or config.auto_highlight) and SPEECH_TO_TEXT_AVAILABLE:
            try:
                self.speech_to_text = SpeechToText()
            except Exception as e:
                console.print(f"[yellow]Could not initialize speech-to-text: {e}[/yellow]")
        
        if (config.auto_highlight or config.trending_effects) and VISUAL_ANALYSIS_AVAILABLE:
            try:
                self.visual_analyzer = VisualAnalyzer()
            except Exception as e:
                console.print(f"[yellow]Could not initialize visual analyzer: {e}[/yellow]")
    
    def __del__(self):
        """Cleanup temporary files."""
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_shorts_video(self, input_path: Path, output_path: Path, 
                           progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Create a TikTok/YouTube Shorts style video from input."""
        
        try:
            if progress_callback:
                progress_callback("Analyzing input video...")
            
            # Get video info
            video_info = self._get_video_info(input_path)
            
            if progress_callback:
                progress_callback("Extracting audio for transcription...")
            
            # Extract and analyze audio for captions
            transcript_segments = []
            if self.config.add_captions and self.speech_to_text:
                transcript_segments = self._extract_transcript(input_path)
            
            if progress_callback:
                progress_callback("Analyzing visual content...")
            
            # Analyze visual content for auto-highlights
            visual_highlights = []
            if self.config.auto_highlight and self.visual_analyzer:
                visual_highlights = self._analyze_visual_highlights(input_path)
            
            if progress_callback:
                progress_callback("Planning video segments and effects...")
            
            # Plan video segments with effects
            video_segments = self._plan_video_segments(
                video_info, transcript_segments, visual_highlights
            )
            
            if progress_callback:
                progress_callback("Generating captions...")
            
            # Generate caption overlays
            caption_segments = []
            if self.config.add_captions and transcript_segments:
                caption_segments = self._generate_captions(transcript_segments)
            
            if progress_callback:
                progress_callback("Applying video transformations...")
            
            # Apply video transformations
            processed_video = self._apply_video_transformations(
                input_path, video_segments, video_info
            )
            
            if progress_callback:
                progress_callback("Adding captions and text overlays...")
            
            # Add captions and text overlays
            if caption_segments:
                processed_video = self._add_captions_to_video(
                    processed_video, caption_segments
                )
            
            if progress_callback:
                progress_callback("Adding trending audio effects...")
            
            # Add audio/music
            if self.config.add_music:
                processed_video = self._add_background_music(processed_video)
            
            if progress_callback:
                progress_callback("Finalizing video...")
            
            # Final processing and export
            result_stats = self._finalize_video(processed_video, output_path)
            
            if progress_callback:
                progress_callback("Complete!")
            
            return {
                "success": True,
                "output_path": str(output_path),
                "duration": result_stats.get("duration"),
                "resolution": f"{self.config.output_resolution[0]}x{self.config.output_resolution[1]}",
                "caption_count": len(caption_segments),
                "effects_applied": [seg.effects for seg in video_segments],
                "stats": result_stats
            }
            
        except Exception as e:
            console.print(f"[red]Error creating shorts video: {e}[/red]")
            return {"success": False, "error": str(e)}
    
    def _get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Extract video metadata using ffprobe."""
        if not FFMPEG_AVAILABLE:
            console.print("[yellow]Using fallback video info (ffmpeg not available)[/yellow]")
            return {
                "duration": 60.0,  # Default
                "width": 1920,
                "height": 1080,
                "fps": 30.0,
                "codec": "unknown",
                "bitrate": 2000000
            }
        
        try:
            probe = ffmpeg.probe(str(video_path))
            video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            
            info = {
                "duration": float(probe['format']['duration']),
                "width": int(video_stream['width']),
                "height": int(video_stream['height']),
                "fps": eval(video_stream['r_frame_rate']),
                "codec": video_stream['codec_name'],
                "bitrate": int(probe['format'].get('bit_rate', 0))
            }
            
            console.print(f"[cyan]Video info: {info['width']}x{info['height']}, {info['duration']:.1f}s, {info['fps']:.1f}fps[/cyan]")
            return info
            
        except Exception as e:
            console.print(f"[red]Error getting video info: {e}[/red]")
            return {}
    
    def _extract_transcript(self, video_path: Path) -> List[TranscriptSegment]:
        """Extract audio and generate transcript."""
        if not self.speech_to_text:
            console.print("[yellow]Speech-to-text not available, skipping transcript[/yellow]")
            return []
        
        if not FFMPEG_AVAILABLE:
            console.print("[yellow]FFmpeg not available, cannot extract audio for transcription[/yellow]")
            return []
        
        try:
            # Extract audio to temporary file
            audio_path = self.temp_dir / "audio.wav"
            
            (
                ffmpeg
                .input(str(video_path))
                .output(str(audio_path), acodec='pcm_s16le', ac=1, ar='16000')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            # Generate transcript
            segments = self.speech_to_text.transcribe_audio(audio_path)
            console.print(f"[green]Generated transcript with {len(segments)} segments[/green]")
            
            return segments
            
        except Exception as e:
            console.print(f"[yellow]Could not extract transcript: {e}[/yellow]")
            return []
    
    def _analyze_visual_highlights(self, video_path: Path) -> List[Dict[str, Any]]:
        """Analyze video for visual highlights and interesting moments."""
        if not self.visual_analyzer:
            console.print("[yellow]Visual analyzer not available, skipping highlight detection[/yellow]")
            return []
        
        if not FFMPEG_AVAILABLE:
            console.print("[yellow]FFmpeg not available, cannot extract frames for analysis[/yellow]")
            return []
        
        try:
            # Extract frames for analysis (every 2 seconds)
            frame_interval = 2.0
            frames_dir = self.temp_dir / "frames"
            frames_dir.mkdir(exist_ok=True)
            
            # Extract frames
            (
                ffmpeg
                .input(str(video_path))
                .filter('fps', f'1/{frame_interval}')
                .output(str(frames_dir / 'frame_%04d.jpg'))
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            # Analyze frames
            frame_paths = sorted(frames_dir.glob('*.jpg'))
            analysis_results = self.visual_analyzer.analyze_frames(frame_paths, 1/frame_interval)
            
            # Extract highlights
            highlights = analysis_results.get('highlights', [])
            console.print(f"[green]Found {len(highlights)} visual highlights[/green]")
            
            return highlights
            
        except Exception as e:
            console.print(f"[yellow]Could not analyze visual highlights: {e}[/yellow]")
            return []
    
    def _plan_video_segments(self, video_info: Dict[str, Any], 
                           transcript_segments: List[TranscriptSegment],
                           visual_highlights: List[Dict[str, Any]]) -> List[VideoSegment]:
        """Plan video segments with appropriate effects and timing."""
        
        duration = min(video_info.get('duration', 60), self.config.target_duration)
        segments = []
        
        # Calculate segment duration (aim for 2-4 second segments for dynamic editing)
        base_segment_duration = 3.0
        num_segments = int(duration / base_segment_duration)
        
        for i in range(num_segments):
            start_time = i * base_segment_duration
            end_time = min(start_time + base_segment_duration, duration)
            
            # Determine effects for this segment
            effects = []
            zoom_factor = 1.0
            speed_factor = 1.0
            
            # Check if this segment has visual highlights
            segment_highlights = [h for h in visual_highlights 
                                if start_time <= h.get('timestamp', 0) <= end_time]
            
            # Check if this segment has exciting transcript content
            segment_transcript = [t for t in transcript_segments
                                if not (t.end < start_time or t.start > end_time)]
            
            # Apply effects based on content analysis
            if segment_highlights:
                if self.config.add_zooms:
                    effects.append("zoom_punch")
                    zoom_factor = self.config.zoom_intensity
                
                if self.config.trending_effects:
                    effects.append("flash_transition")
            
            # Check for exciting speech content
            exciting_words = ['amazing', 'incredible', 'wow', 'unbelievable', 'awesome']
            has_exciting_speech = any(
                any(word in seg.text.lower() for word in exciting_words)
                for seg in segment_transcript
            )
            
            if has_exciting_speech and self.config.speed_variations:
                effects.append("speed_ramp")
                speed_factor = self.config.speed_factor
            
            # Add text overlay if there's good content
            text_overlay = None
            if segment_transcript and len(segment_transcript[0].text.split()) <= 8:
                text_overlay = segment_transcript[0].text
            
            segment = VideoSegment(
                start_time=start_time,
                end_time=end_time,
                effects=effects,
                zoom_factor=zoom_factor,
                speed_factor=speed_factor,
                text_overlay=text_overlay
            )
            
            segments.append(segment)
        
        console.print(f"[cyan]Planned {len(segments)} video segments with effects[/cyan]")
        return segments
    
    def _generate_captions(self, transcript_segments: List[TranscriptSegment]) -> List[CaptionSegment]:
        """Generate styled captions from transcript."""
        caption_segments = []
        colors = self.config.get_current_color_scheme()
        
        # Calculate position based on config
        width, height = self.config.output_resolution
        
        if self.config.caption_position == "bottom":
            y_pos = height - self.config.caption_margin
        elif self.config.caption_position == "top":
            y_pos = self.config.caption_margin
        else:  # center
            y_pos = height // 2
        
        x_pos = width // 2  # Center horizontally
        
        for segment in transcript_segments:
            # Split long text into multiple lines
            words = segment.text.split()
            lines = []
            current_line = []
            
            # Aim for 3-5 words per line for readability
            for word in words:
                current_line.append(word)
                if len(current_line) >= 4:
                    lines.append(' '.join(current_line))
                    current_line = []
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Create caption with styling
            text = '\\n'.join(lines)  # FFmpeg subtitle format
            
            style = {
                "font_size": self.config.caption_font_size,
                "font_color": colors["text_color"],
                "background_color": colors["background_color"],
                "border_color": colors["accent_color"],
                "font_name": self.config.caption_font
            }
            
            caption = CaptionSegment(
                text=text,
                start_time=segment.start,
                end_time=segment.end,
                style=style,
                position=(x_pos, y_pos)
            )
            
            caption_segments.append(caption)
        
        console.print(f"[green]Generated {len(caption_segments)} caption segments[/green]")
        return caption_segments
    
    def _apply_video_transformations(self, input_path: Path, 
                                   segments: List[VideoSegment],
                                   video_info: Dict[str, Any]) -> Path:
        """Apply zooms, speed changes, and format conversion."""
        
        if not FFMPEG_AVAILABLE:
            console.print("[yellow]FFmpeg not available, using simple copy fallback[/yellow]")
            # Simple copy operation
            output_path = self.temp_dir / "transformed.mp4"
            import shutil
            shutil.copy2(input_path, output_path)
            return output_path
        
        output_path = self.temp_dir / "transformed.mp4"
        width, height = self.config.output_resolution
        
        # Build complex FFmpeg filter for transformations
        inputs = [ffmpeg.input(str(input_path))]
        
        # Create filter chain for each segment
        filter_chains = []
        
        for i, segment in enumerate(segments):
            # Extract segment
            segment_input = inputs[0].video.filter(
                'trim', 
                start=segment.start_time, 
                end=segment.end_time
            ).filter('setpts', 'PTS-STARTPTS')
            
            # Apply speed change
            if segment.speed_factor != 1.0:
                segment_input = segment_input.filter(
                    'setpts', 
                    f'{1/segment.speed_factor}*PTS'
                )
            
            # Apply zoom effect
            if segment.zoom_factor != 1.0:
                # Calculate zoom parameters
                zoom_scale = segment.zoom_factor
                segment_input = segment_input.filter(
                    'scale', 
                    f'iw*{zoom_scale}', f'ih*{zoom_scale}'
                ).filter(
                    'crop', 
                    f'iw/{zoom_scale}', f'ih/{zoom_scale}'
                )
            
            # Scale to target resolution
            segment_input = segment_input.filter(
                'scale', width, height, force_original_aspect_ratio='increase'
            ).filter(
                'crop', width, height
            )
            
            filter_chains.append(segment_input)
        
        # Concatenate all segments
        if filter_chains:
            concatenated = ffmpeg.concat(*filter_chains, v=1, a=0)
        else:
            # Fallback: just scale the original video
            concatenated = inputs[0].video.filter(
                'scale', width, height, force_original_aspect_ratio='increase'
            ).filter(
                'crop', width, height
            )
        
        # Add audio back (scaled for speed changes if needed)
        audio = inputs[0].audio
        
        # Output with high quality settings
        output = ffmpeg.output(
            concatenated, audio,
            str(output_path),
            vcodec='libx264',
            acodec='aac',
            preset='medium',
            crf=23,
            r=self.config.fps,
            video_bitrate=self.config.bitrate
        ).overwrite_output()
        
        # Run FFmpeg
        try:
            ffmpeg.run(output, capture_stdout=True, capture_stderr=True)
            console.print(f"[green]Applied video transformations[/green]")
            return output_path
        except ffmpeg.Error as e:
            console.print(f"[red]FFmpeg error: {e.stderr.decode()}[/red]")
            raise
    
    def _add_captions_to_video(self, video_path: Path, 
                             captions: List[CaptionSegment]) -> Path:
        """Add styled captions to video."""
        
        if not FFMPEG_AVAILABLE:
            console.print("[yellow]FFmpeg not available, skipping caption overlay[/yellow]")
            return video_path
        
        output_path = self.temp_dir / "with_captions.mp4"
        
        # Generate subtitle file
        srt_path = self.temp_dir / "captions.srt"
        self._generate_srt_file(captions, srt_path)
        
        # Apply captions using FFmpeg
        input_video = ffmpeg.input(str(video_path))
        
        # Get caption style
        colors = self.config.get_current_color_scheme()
        
        # Build subtitle filter
        subtitle_filter = (
            f"subtitles={srt_path}:"
            f"force_style='FontName={self.config.caption_font},"
            f"FontSize={self.config.caption_font_size},"
            f"PrimaryColour={self._hex_to_bgr(colors['text_color'])},"
            f"BackColour={self._hex_to_bgr(colors['background_color'])},"
            f"Bold=1,Alignment=2'"
        )
        
        output = ffmpeg.output(
            input_video.video.filter('subtitles', str(srt_path)),
            input_video.audio,
            str(output_path),
            vcodec='libx264',
            acodec='copy'
        ).overwrite_output()
        
        try:
            ffmpeg.run(output, capture_stdout=True, capture_stderr=True)
            console.print(f"[green]Added captions to video[/green]")
            return output_path
        except ffmpeg.Error as e:
            console.print(f"[red]Error adding captions: {e.stderr.decode()}[/red]")
            return video_path
    
    def _generate_srt_file(self, captions: List[CaptionSegment], output_path: Path):
        """Generate SRT subtitle file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, caption in enumerate(captions, 1):
                start_time = self._seconds_to_srt_time(caption.start_time)
                end_time = self._seconds_to_srt_time(caption.end_time)
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{caption.text}\n\n")
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')
    
    def _hex_to_bgr(self, hex_color: str) -> str:
        """Convert hex color to BGR format for FFmpeg."""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f"&H00{b:02x}{g:02x}{r:02x}"
    
    def _add_background_music(self, video_path: Path) -> Path:
        """Add trending background music."""
        # This would integrate with music library
        # For now, just return the original video
        console.print("[yellow]Music integration not implemented yet[/yellow]")
        return video_path
    
    def _finalize_video(self, video_path: Path, output_path: Path) -> Dict[str, Any]:
        """Final video processing and export."""
        
        # Copy the final video to output location
        shutil.copy2(video_path, output_path)
        
        # Get final video stats
        final_info = self._get_video_info(output_path)
        
        console.print(f"[green]✅ Shorts video created: {output_path}[/green]")
        console.print(f"[cyan]Final specs: {final_info.get('width')}x{final_info.get('height')}, {final_info.get('duration', 0):.1f}s[/cyan]")
        
        return final_info
    
    def generate_preview(self, input_path: Path, effect_type: str, duration: float) -> Path:
        """Generate a preview of specific effects."""
        # Implementation for preview generation
        preview_path = self.temp_dir / f"preview_{effect_type}.mp4"
        
        # This would create a short preview showing the requested effects
        console.print(f"[blue]Generating {effect_type} preview...[/blue]")
        
        return preview_path