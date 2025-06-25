"""
TikTok/YouTube Shorts Video Editor using MoviePy
Creates mobile-optimized viral videos with captions, effects, and trending elements.
"""

try:
    from moviepy.editor import (
        VideoFileClip, AudioFileClip, TextClip, ColorClip, 
        CompositeVideoClip, concatenate_videoclips
    )
    import moviepy.video.fx.all as vfx
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    # Create dummy classes for type hints
    class VideoFileClip: pass
    class AudioFileClip: pass
    class TextClip: pass
    class ColorClip: pass
    class CompositeVideoClip: pass

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Minimal numpy substitute
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
        
        @staticmethod
        def clip(arr, min_val, max_val):
            return [max(min_val, min(max_val, x)) for x in arr]

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
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
    position: str = "bottom"

class MoviePyShortsEditor:
    """MoviePy-based video editor for creating TikTok/YouTube Shorts style content."""
    
    def __init__(self, config: ShortsConfig):
        self.config = config
        self.temp_dir = Path(tempfile.mkdtemp(prefix="moviepy_shorts_"))
        self.speech_to_text = None
        self.visual_analyzer = None
        
        # Check dependencies
        if not MOVIEPY_AVAILABLE:
            console.print("[red]Error: MoviePy not available. Install with: pip install moviepy[/red]")
            raise ImportError("MoviePy is required for video processing")
        
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
        """Create a TikTok/YouTube Shorts style video from input using MoviePy."""
        
        try:
            if progress_callback:
                progress_callback("Loading video with MoviePy...")
            
            # Load video
            video = VideoFileClip(str(input_path))
            original_duration = video.duration
            
            console.print(f"[cyan]Loaded video: {video.w}x{video.h}, {original_duration:.1f}s, {video.fps}fps[/cyan]")
            
            if progress_callback:
                progress_callback("Analyzing content for optimal segments...")
            
            # Get transcript if available
            transcript_segments = []
            if self.config.add_captions and self.speech_to_text:
                transcript_segments = self._extract_transcript_moviepy(video)
            
            # Analyze visual highlights if available
            visual_highlights = []
            if self.config.auto_highlight and self.visual_analyzer:
                visual_highlights = self._analyze_visual_highlights_moviepy(video)
            
            if progress_callback:
                progress_callback("Planning dynamic segments and effects...")
            
            # Plan video segments
            video_segments = self._plan_video_segments_moviepy(
                video, transcript_segments, visual_highlights
            )
            
            if progress_callback:
                progress_callback("Applying mobile optimization and effects...")
            
            # Create the shorts video
            shorts_video = self._create_shorts_from_segments(video, video_segments)
            
            if progress_callback:
                progress_callback("Adding captions and text overlays...")
            
            # Add captions if available
            if self.config.add_captions and transcript_segments:
                caption_segments = self._generate_captions_moviepy(transcript_segments)
                shorts_video = self._add_captions_moviepy(shorts_video, caption_segments)
            
            if progress_callback:
                progress_callback("Finalizing and exporting...")
            
            # Export the final video
            result_stats = self._export_video_moviepy(shorts_video, output_path)
            
            # Cleanup
            video.close()
            if 'shorts_video' in locals():
                shorts_video.close()
            
            if progress_callback:
                progress_callback("Complete!")
            
            return {
                "success": True,
                "output_path": str(output_path),
                "duration": result_stats.get("duration"),
                "resolution": f"{self.config.output_resolution[0]}x{self.config.output_resolution[1]}",
                "caption_count": len(transcript_segments) if transcript_segments else 0,
                "effects_applied": [seg.effects for seg in video_segments],
                "stats": result_stats
            }
            
        except Exception as e:
            console.print(f"[red]Error creating shorts video: {e}[/red]")
            return {"success": False, "error": str(e)}
    
    def _extract_transcript_moviepy(self, video: VideoFileClip) -> List[TranscriptSegment]:
        """Extract audio and generate transcript using MoviePy."""
        if not self.speech_to_text:
            console.print("[yellow]Speech-to-text not available[/yellow]")
            return []
        
        try:
            # Extract audio using MoviePy
            audio_path = self.temp_dir / "audio.wav"
            video.audio.write_audiofile(str(audio_path), logger=None, verbose=False)
            
            # Generate transcript
            segments = self.speech_to_text.transcribe_audio(audio_path)
            console.print(f"[green]Generated transcript with {len(segments)} segments[/green]")
            
            return segments
            
        except Exception as e:
            console.print(f"[yellow]Could not extract transcript: {e}[/yellow]")
            return []
    
    def _analyze_visual_highlights_moviepy(self, video: VideoFileClip) -> List[Dict[str, Any]]:
        """Analyze video for visual highlights using MoviePy."""
        if not self.visual_analyzer:
            console.print("[yellow]Visual analyzer not available[/yellow]")
            return []
        
        try:
            # Extract frames every 2 seconds for analysis
            frame_interval = 2.0
            frames_dir = self.temp_dir / "frames"
            frames_dir.mkdir(exist_ok=True)
            
            highlights = []
            for t in np.arange(0, min(video.duration, 60), frame_interval):  # Limit to first 60s for speed
                frame_path = frames_dir / f"frame_{t:.1f}s.jpg"
                frame = video.get_frame(t)
                
                # Convert to PIL image and save
                from PIL import Image
                Image.fromarray(frame).save(frame_path)
                
                # Simple highlight detection based on frame variance
                frame_variance = np.var(frame)
                if frame_variance > 1000:  # Threshold for "interesting" frames
                    highlights.append({
                        'timestamp': t,
                        'score': min(frame_variance / 10000, 1.0),
                        'type': 'visual'
                    })
            
            console.print(f"[green]Found {len(highlights)} visual highlights[/green]")
            return highlights
            
        except Exception as e:
            console.print(f"[yellow]Could not analyze visual highlights: {e}[/yellow]")
            return []
    
    def _plan_video_segments_moviepy(self, video: VideoFileClip, 
                                   transcript_segments: List[TranscriptSegment],
                                   visual_highlights: List[Dict[str, Any]]) -> List[VideoSegment]:
        """Plan video segments with appropriate effects using MoviePy."""
        
        target_duration = min(video.duration, self.config.target_duration)
        segments = []
        
        # Calculate segment duration for dynamic editing
        base_segment_duration = 3.0  # 3-second segments for TikTok style
        num_segments = int(target_duration / base_segment_duration)
        
        for i in range(num_segments):
            start_time = i * base_segment_duration
            end_time = min(start_time + base_segment_duration, target_duration)
            
            # Determine effects for this segment
            effects = []
            zoom_factor = 1.0
            speed_factor = 1.0
            
            # Check for visual highlights in this segment
            segment_highlights = [h for h in visual_highlights 
                                if start_time <= h.get('timestamp', 0) <= end_time]
            
            # Check for exciting transcript content
            segment_transcript = [t for t in transcript_segments
                                if not (t.end < start_time or t.start > end_time)]
            
            # Apply effects based on content analysis
            if segment_highlights and self.config.add_zooms:
                effects.append("zoom_punch")
                zoom_factor = self.config.zoom_intensity
            
            # Check for exciting speech content
            exciting_words = ['amazing', 'incredible', 'wow', 'unbelievable', 'awesome', 'fantastic']
            has_exciting_speech = any(
                any(word in seg.text.lower() for word in exciting_words)
                for seg in segment_transcript
            )
            
            if has_exciting_speech and self.config.speed_variations:
                effects.append("speed_ramp")
                speed_factor = self.config.speed_factor
            
            # Add text overlay for short, impactful text
            text_overlay = None
            if segment_transcript and len(segment_transcript[0].text.split()) <= 6:
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
    
    def _create_shorts_from_segments(self, video: VideoFileClip, 
                                   segments: List[VideoSegment]) -> VideoFileClip:
        """Create the shorts video by applying effects to segments."""
        
        processed_clips = []
        
        for segment in segments:
            # Extract segment
            clip = video.subclip(segment.start_time, segment.end_time)
            
            # Apply speed effect
            if segment.speed_factor != 1.0:
                clip = clip.fx(vfx.speedx, segment.speed_factor)
            
            # Apply zoom effect
            if segment.zoom_factor != 1.0:
                zoom = segment.zoom_factor
                clip = clip.fx(vfx.resize, zoom).fx(vfx.crop, 
                    width=clip.w/zoom, height=clip.h/zoom, 
                    x_center=clip.w/2, y_center=clip.h/2)
            
            # Add visual effects based on effect list
            if "flash_transition" in segment.effects:
                # Add a brief white flash at the beginning
                flash = ColorClip(size=clip.size, color=(255, 255, 255), duration=0.1)
                clip = concatenate_videoclips([flash, clip])
            
            processed_clips.append(clip)
        
        # Concatenate all segments
        if processed_clips:
            final_video = concatenate_videoclips(processed_clips)
        else:
            # Fallback: just resize the original video
            final_video = video.subclip(0, min(video.duration, self.config.target_duration))
        
        # Resize to target resolution for mobile
        target_width, target_height = self.config.output_resolution
        
        # Calculate scaling to fit the target aspect ratio
        current_aspect = final_video.w / final_video.h
        target_aspect = target_width / target_height
        
        if current_aspect > target_aspect:
            # Video is wider, scale by height and crop width
            new_height = target_height
            new_width = int(final_video.w * target_height / final_video.h)
            final_video = final_video.fx(vfx.resize, width=new_width, height=new_height)
            final_video = final_video.fx(vfx.crop, width=target_width, height=target_height,
                                       x_center=new_width//2, y_center=new_height//2)
        else:
            # Video is taller, scale by width and crop height
            new_width = target_width
            new_height = int(final_video.h * target_width / final_video.w)
            final_video = final_video.fx(vfx.resize, width=new_width, height=new_height)
            final_video = final_video.fx(vfx.crop, width=target_width, height=target_height,
                                       x_center=new_width//2, y_center=new_height//2)
        
        console.print(f"[green]Created shorts video: {final_video.w}x{final_video.h}, {final_video.duration:.1f}s[/green]")
        return final_video
    
    def _generate_captions_moviepy(self, transcript_segments: List[TranscriptSegment]) -> List[CaptionSegment]:
        """Generate styled captions from transcript for MoviePy."""
        caption_segments = []
        
        for segment in transcript_segments:
            # Split long text into readable chunks
            words = segment.text.split()
            lines = []
            current_line = []
            
            for word in words:
                current_line.append(word)
                if len(current_line) >= 4:  # 4 words per line for readability
                    lines.append(' '.join(current_line))
                    current_line = []
            
            if current_line:
                lines.append(' '.join(current_line))
            
            text = '\n'.join(lines)
            
            # Get styling from config
            colors = self.config.get_current_color_scheme()
            style = {
                "fontsize": self.config.caption_font_size,
                "color": colors["text_color"],
                "font": self.config.caption_font,
                "stroke_color": colors["background_color"],
                "stroke_width": 3
            }
            
            caption = CaptionSegment(
                text=text,
                start_time=segment.start,
                end_time=segment.end,
                style=style,
                position=self.config.caption_position
            )
            
            caption_segments.append(caption)
        
        console.print(f"[green]Generated {len(caption_segments)} caption segments[/green]")
        return caption_segments
    
    def _add_captions_moviepy(self, video: VideoFileClip, 
                            captions: List[CaptionSegment]) -> VideoFileClip:
        """Add styled captions to video using MoviePy."""
        
        try:
            text_clips = []
            
            for caption in captions:
                # Calculate position
                if caption.position == "bottom":
                    pos = ('center', video.h - self.config.caption_margin)
                elif caption.position == "top":
                    pos = ('center', self.config.caption_margin)
                else:  # center
                    pos = ('center', 'center')
                
                # Create text clip
                txt_clip = TextClip(
                    caption.text,
                    fontsize=caption.style["fontsize"],
                    color=caption.style["color"],
                    font=caption.style.get("font", "Arial-Bold"),
                    stroke_color=caption.style.get("stroke_color", "black"),
                    stroke_width=caption.style.get("stroke_width", 2)
                ).set_position(pos).set_start(caption.start_time).set_end(caption.end_time)
                
                text_clips.append(txt_clip)
            
            # Composite video with text
            if text_clips:
                final_video = CompositeVideoClip([video] + text_clips)
                console.print(f"[green]Added {len(text_clips)} caption overlays[/green]")
                return final_video
            else:
                return video
                
        except Exception as e:
            console.print(f"[yellow]Could not add captions: {e}[/yellow]")
            return video
    
    def _export_video_moviepy(self, video: VideoFileClip, output_path: Path) -> Dict[str, Any]:
        """Export the final video using MoviePy."""
        
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Export with high quality settings
            video.write_videofile(
                str(output_path),
                fps=self.config.fps,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile=str(self.temp_dir / 'temp_audio.m4a'),
                remove_temp=True,
                logger=None,  # Suppress moviepy logs
                verbose=False
            )
            
            # Get final stats
            stats = {
                "duration": video.duration,
                "width": video.w,
                "height": video.h,
                "fps": video.fps
            }
            
            console.print(f"[green]✅ Exported shorts video: {output_path}[/green]")
            console.print(f"[cyan]Final specs: {stats['width']}x{stats['height']}, {stats['duration']:.1f}s[/cyan]")
            
            return stats
            
        except Exception as e:
            console.print(f"[red]Export error: {e}[/red]")
            raise