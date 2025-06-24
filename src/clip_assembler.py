"""Clip assembly and post-processing for viral video generation."""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import subprocess
import tempfile
from rich.console import Console
from rich.progress import Progress, track

from .config import config
from .video_processor import VideoProcessor
from .virality_scorer import ViralMoment
from .speech_to_text import TranscriptSegment

console = Console()

@dataclass
class ClipStyle:
    """Defines the visual style for a viral clip."""
    resolution: Tuple[int, int] = (1920, 1080)
    fps: int = 30
    subtitle_font_size: int = 48
    subtitle_color: str = "white"
    subtitle_outline_color: str = "black"
    subtitle_outline_width: int = 2
    fade_in_duration: float = 0.5
    fade_out_duration: float = 0.5
    zoom_effect: bool = True
    zoom_intensity: float = 0.1
    color_correction: bool = True
    audio_enhancement: bool = True

class ClipAssembler:
    """Assemble and post-process viral clips from analyzed moments."""
    
    def __init__(self, output_dir: Optional[Path] = None, temp_dir: Optional[Path] = None):
        self.output_dir = output_dir or config.OUTPUTS_DIR
        self.temp_dir = temp_dir or config.TEMP_DIR
        self.video_processor = VideoProcessor(self.temp_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default clip style
        self.default_style = ClipStyle()
    
    def create_viral_clip(self, video_path: Path, viral_moment: ViralMoment,
                         style: Optional[ClipStyle] = None,
                         output_path: Optional[Path] = None) -> Optional[Path]:
        """Create a viral clip from a video and viral moment."""
        
        style = style or self.default_style
        
        if not output_path:
            safe_filename = f"viral_clip_{viral_moment.start_time:.1f}s_{viral_moment.virality_score:.2f}.mp4"
            output_path = self.output_dir / safe_filename
        
        console.print(f"[blue]Creating viral clip: {output_path.name}[/blue]")
        
        try:
            # Step 1: Extract base clip
            base_clip = self._extract_base_clip(video_path, viral_moment, style)
            if not base_clip:
                return None
            
            # Step 2: Add visual effects
            enhanced_clip = self._add_visual_effects(base_clip, style)
            if not enhanced_clip:
                enhanced_clip = base_clip
            
            # Step 3: Add subtitles
            subtitled_clip = self._add_subtitles(enhanced_clip, viral_moment, style)
            if not subtitled_clip:
                subtitled_clip = enhanced_clip
            
            # Step 4: Enhance audio
            final_clip = self._enhance_audio(subtitled_clip, style)
            if not final_clip:
                final_clip = subtitled_clip
            
            # Step 5: Final processing and move to output
            if final_clip != output_path:
                final_processed = self._final_processing(final_clip, output_path, style)
                if final_processed:
                    console.print(f"[green]Viral clip created: {output_path}[/green]")
                    return output_path
            
            return final_clip
            
        except Exception as e:
            console.print(f"[red]Error creating viral clip: {e}[/red]")
            return None
    
    def _extract_base_clip(self, video_path: Path, viral_moment: ViralMoment, 
                          style: ClipStyle) -> Optional[Path]:
        """Extract the base clip from the video."""
        
        # Add some padding to the clip for better context
        padding = 2.0  # 2 seconds padding
        start_time = max(0, viral_moment.start_time - padding)
        end_time = viral_moment.end_time + padding
        duration = end_time - start_time
        
        # Ensure duration is within limits
        if duration > config.MAX_CLIP_DURATION:
            excess = duration - config.MAX_CLIP_DURATION
            start_time += excess / 2
            end_time -= excess / 2
            duration = config.MAX_CLIP_DURATION
        
        output_path = self.temp_dir / f"base_clip_{viral_moment.start_time:.1f}s.mp4"
        
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-ss', str(start_time),
            '-t', str(duration),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-preset', 'medium',
            '-crf', '20',
            '-y', str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return output_path
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Base clip extraction failed: {e}[/red]")
            return None
    
    def _add_visual_effects(self, video_path: Path, style: ClipStyle) -> Optional[Path]:
        """Add visual effects like zoom, color correction, and transitions."""
        
        output_path = self.temp_dir / f"enhanced_{video_path.name}"
        
        # Build filter chain
        filters = []
        
        # Color correction
        if style.color_correction:
            filters.append("eq=contrast=1.1:brightness=0.05:saturation=1.2")
        
        # Zoom effect for engagement
        if style.zoom_effect:
            # Subtle zoom in effect
            zoom_filter = f"scale=iw*(1+{style.zoom_intensity}*t/duration):ih*(1+{style.zoom_intensity}*t/duration)"
            filters.append(zoom_filter)
        
        # Fade effects
        if style.fade_in_duration > 0:
            filters.append(f"fade=t=in:st=0:d={style.fade_in_duration}")
        
        if style.fade_out_duration > 0:
            # We'll calculate the fade out start time during processing
            filters.append(f"fade=t=out:d={style.fade_out_duration}")
        
        # Combine filters
        if filters:
            filter_string = ",".join(filters)
            
            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-vf', filter_string,
                '-c:a', 'copy',
                '-y', str(output_path)
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                return output_path
            except subprocess.CalledProcessError as e:
                console.print(f"[yellow]Visual effects failed: {e}[/yellow]")
                return None
        
        return video_path  # Return original if no effects
    
    def _add_subtitles(self, video_path: Path, viral_moment: ViralMoment, 
                      style: ClipStyle) -> Optional[Path]:
        """Add animated subtitles to the clip."""
        
        if not viral_moment.transcript_segments:
            return video_path
        
        output_path = self.temp_dir / f"subtitled_{video_path.name}"
        
        # Create subtitle file
        subtitle_file = self._create_subtitle_file(viral_moment.transcript_segments, style)
        if not subtitle_file:
            return video_path
        
        # Add subtitles with custom styling
        subtitle_filter = (
            f"subtitles='{subtitle_file}':force_style='"
            f"FontName=Arial Bold,FontSize={style.subtitle_font_size},"
            f"PrimaryColour=&H{self._color_to_hex(style.subtitle_color)},"
            f"OutlineColour=&H{self._color_to_hex(style.subtitle_outline_color)},"
            f"Outline={style.subtitle_outline_width},"
            f"Alignment=2'"  # Bottom center alignment
        )
        
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vf', subtitle_filter,
            '-c:a', 'copy',
            '-y', str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return output_path
        except subprocess.CalledProcessError as e:
            console.print(f"[yellow]Subtitle addition failed: {e}[/yellow]")
            return video_path
    
    def _create_subtitle_file(self, segments: List[TranscriptSegment], 
                            style: ClipStyle) -> Optional[Path]:
        """Create SRT subtitle file from transcript segments."""
        
        subtitle_file = self.temp_dir / "subtitles.srt"
        
        try:
            with open(subtitle_file, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(segments, 1):
                    start_time = self._format_srt_time(segment.start)
                    end_time = self._format_srt_time(segment.end)
                    
                    # Clean and format text for better readability
                    text = segment.text.strip()
                    text = self._format_subtitle_text(text)
                    
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")
            
            return subtitle_file
            
        except Exception as e:
            console.print(f"[red]Error creating subtitle file: {e}[/red]")
            return None
    
    def _format_subtitle_text(self, text: str) -> str:
        """Format subtitle text for better readability."""
        # Split long sentences
        if len(text) > 60:
            words = text.split()
            mid_point = len(words) // 2
            line1 = " ".join(words[:mid_point])
            line2 = " ".join(words[mid_point:])
            return f"{line1}\n{line2}"
        
        return text
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format seconds to SRT time format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def _color_to_hex(self, color_name: str) -> str:
        """Convert color name to hex for FFmpeg."""
        color_map = {
            'white': 'FFFFFF',
            'black': '000000',
            'red': 'FF0000',
            'green': '00FF00',
            'blue': '0000FF',
            'yellow': 'FFFF00',
            'cyan': '00FFFF',
            'magenta': 'FF00FF'
        }
        return color_map.get(color_name.lower(), 'FFFFFF')
    
    def _enhance_audio(self, video_path: Path, style: ClipStyle) -> Optional[Path]:
        """Enhance audio quality and add effects."""
        
        if not style.audio_enhancement:
            return video_path
        
        output_path = self.temp_dir / f"audio_enhanced_{video_path.name}"
        
        # Audio enhancement filters
        audio_filters = [
            "loudnorm=I=-16:TP=-1.5:LRA=11",  # Normalize loudness
            "highpass=f=80",  # Remove low-frequency noise
            "lowpass=f=15000",  # Remove high-frequency noise
            "compand=attacks=0.3:decays=0.8:points=-80/-80|-45/-45|-27/-25|0/-7:soft-knee=0.05:gain=5"  # Dynamic compression
        ]
        
        audio_filter_string = ",".join(audio_filters)
        
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-af', audio_filter_string,
            '-c:v', 'copy',
            '-y', str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return output_path
        except subprocess.CalledProcessError as e:
            console.print(f"[yellow]Audio enhancement failed: {e}[/yellow]")
            return video_path
    
    def _final_processing(self, video_path: Path, output_path: Path, 
                         style: ClipStyle) -> Optional[Path]:
        """Final processing and optimization for social media."""
        
        # Get target resolution
        width, height = style.resolution
        
        # Optimize for social media
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-maxrate', '4M',
            '-bufsize', '8M',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-r', str(style.fps),
            '-movflags', '+faststart',  # Optimize for streaming
            '-y', str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            console.print(f"[green]Final processing completed: {output_path}[/green]")
            return output_path
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Final processing failed: {e}[/red]")
            return None
    
    def create_thumbnail(self, video_path: Path, viral_moment: ViralMoment,
                        output_path: Optional[Path] = None) -> Optional[Path]:
        """Create an eye-catching thumbnail for the viral clip."""
        
        if not output_path:
            output_path = self.output_dir / f"thumbnail_{viral_moment.start_time:.1f}s.jpg"
        
        # Use the middle of the viral moment for thumbnail
        thumbnail_time = viral_moment.start_time + (viral_moment.duration / 2)
        
        # Extract frame
        temp_frame = self.temp_dir / "temp_thumbnail.jpg"
        
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-ss', str(thumbnail_time),
            '-vframes', '1',
            '-vf', 'scale=1280:720',
            '-q:v', '2',
            '-y', str(temp_frame)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Enhance thumbnail
            enhanced_thumbnail = self._enhance_thumbnail(temp_frame, viral_moment)
            
            if enhanced_thumbnail:
                # Move to final location
                import shutil
                shutil.move(str(enhanced_thumbnail), str(output_path))
                console.print(f"[green]Thumbnail created: {output_path}[/green]")
                return output_path
            
            return None
            
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Thumbnail creation failed: {e}[/red]")
            return None
    
    def _enhance_thumbnail(self, image_path: Path, viral_moment: ViralMoment) -> Optional[Path]:
        """Enhance thumbnail with overlays and effects."""
        
        try:
            # Load image
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)
            
            # Add score indicator
            score_text = f"🔥 {viral_moment.virality_score:.1f}"
            
            # Try to load font
            try:
                font = ImageFont.truetype("arial.ttf", 60)
                small_font = ImageFont.truetype("arial.ttf", 40)
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Add score overlay
            text_bbox = draw.textbbox((0, 0), score_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Position in top-right corner
            x = img.width - text_width - 20
            y = 20
            
            # Add shadow
            draw.text((x+2, y+2), score_text, font=font, fill=(0, 0, 0, 180))
            # Add main text
            draw.text((x, y), score_text, font=font, fill=(255, 255, 255, 255))
            
            # Add duration indicator
            duration_text = f"{viral_moment.duration:.1f}s"
            draw.text((x+2, y+text_height+12), duration_text, font=small_font, fill=(0, 0, 0, 180))
            draw.text((x, y+text_height+10), duration_text, font=small_font, fill=(255, 255, 255, 255))
            
            # Save enhanced thumbnail
            enhanced_path = self.temp_dir / f"enhanced_{image_path.name}"
            img.save(enhanced_path, "JPEG", quality=95)
            
            return enhanced_path
            
        except Exception as e:
            console.print(f"[yellow]Thumbnail enhancement failed: {e}[/yellow]")
            return image_path
    
    def create_multiple_clips(self, video_path: Path, viral_moments: List[ViralMoment],
                            style: Optional[ClipStyle] = None) -> List[Path]:
        """Create multiple viral clips from a list of moments."""
        
        clips = []
        
        console.print(f"[blue]Creating {len(viral_moments)} viral clips[/blue]")
        
        for i, moment in enumerate(track(viral_moments, description="Creating clips")):
            clip_path = self.create_viral_clip(video_path, moment, style)
            if clip_path:
                clips.append(clip_path)
                
                # Also create thumbnail
                self.create_thumbnail(video_path, moment)
        
        console.print(f"[green]Created {len(clips)} viral clips successfully[/green]")
        return clips
    
    def create_tiktok_viral_clip(self, video_path: Path, viral_moments: List[ViralMoment],
                                target_duration: float = 90.0,
                                style: Optional[ClipStyle] = None,
                                output_path: Optional[Path] = None) -> Optional[Path]:
        """Create a TikTok-style viral clip with rapid cuts and high engagement."""
        
        if not viral_moments:
            console.print("[yellow]No viral moments to create clip from[/yellow]")
            return None
        
        style = style or self.default_style
        
        if not output_path:
            output_path = self.output_dir / f"tiktok_viral_{target_duration:.0f}s.mp4"
        
        console.print(f"[blue]Creating TikTok-style viral clip ({target_duration}s) from {len(viral_moments)} moments[/blue]")
        
        try:
            # Create rapid-fire segments with optimal pacing
            viral_segments = self._create_viral_segments(viral_moments, target_duration)
            
            if not viral_segments:
                console.print("[yellow]Could not create viral segments[/yellow]")
                return None
            
            console.print(f"[blue]Processing {len(viral_segments)} viral segments...[/blue]")
            
            # Extract all segments
            segment_clips = []
            for i, segment in enumerate(viral_segments):
                console.print(f"[blue]Extracting segment {i+1}/{len(viral_segments)}: {segment['start_time']:.1f}s-{segment['end_time']:.1f}s[/blue]")
                clip = self._extract_viral_segment(video_path, segment, style, i)
                if clip:
                    segment_clips.append(clip)
                else:
                    console.print(f"[yellow]Segment {i+1} extraction failed[/yellow]")
            
            if not segment_clips:
                console.print("[red]No segments were extracted[/red]")
                return None
            
            console.print(f"[blue]Successfully extracted {len(segment_clips)} segments, creating compilation...[/blue]")
            
            # Create viral compilation with rapid cuts and effects
            viral_compilation = self._create_viral_compilation(segment_clips, style)
            if not viral_compilation:
                console.print("[red]Failed to create viral compilation[/red]")
                return None
            
            console.print(f"[blue]Compilation created, applying final effects...[/blue]")
            
            # Final viral post-processing
            final_clip = self._apply_viral_effects(viral_compilation, style, output_path)
            if final_clip:
                # Verify final output
                if output_path.exists() and output_path.stat().st_size > 1024:
                    console.print(f"[green]TikTok viral clip created: {output_path} ({output_path.stat().st_size/1024/1024:.1f}MB)[/green]")
                    return output_path
                else:
                    console.print(f"[red]Final clip is empty or missing: {output_path}[/red]")
                    return None
            
            console.print("[red]Final effects processing failed[/red]")
            return None
            
        except Exception as e:
            console.print(f"[red]Error creating TikTok viral clip: {e}[/red]")
            return None

    def create_single_long_clip(self, video_path: Path, viral_moments: List[ViralMoment],
                               target_duration: float = 90.0,
                               style: Optional[ClipStyle] = None,
                               output_path: Optional[Path] = None) -> Optional[Path]:
        """Create a single long clip by concatenating the best viral moments."""
        
        if not viral_moments:
            console.print("[yellow]No viral moments to create clip from[/yellow]")
            return None
        
        style = style or self.default_style
        
        if not output_path:
            output_path = self.output_dir / f"viral_highlight_{target_duration:.0f}s.mp4"
        
        console.print(f"[blue]Creating single {target_duration}s viral clip from {len(viral_moments)} moments[/blue]")
        
        try:
            # Select and adjust moments to fit target duration
            selected_moments = self._select_moments_for_duration(viral_moments, target_duration)
            
            if not selected_moments:
                console.print("[yellow]Could not select suitable moments for target duration[/yellow]")
                return None
            
            # Create individual clip segments
            clip_segments = []
            for i, moment in enumerate(selected_moments):
                segment_path = self._extract_base_clip(video_path, moment, style)
                if segment_path:
                    clip_segments.append(segment_path)
                else:
                    console.print(f"[yellow]Failed to extract segment {i+1}[/yellow]")
            
            if not clip_segments:
                console.print("[red]No clip segments were created[/red]")
                return None
            
            # Concatenate all segments
            final_clip = self._concatenate_clips_with_transitions(clip_segments, style)
            if not final_clip:
                console.print("[red]Failed to concatenate clips[/red]")
                return None
            
            # Add overall enhancements
            enhanced_clip = self._add_visual_effects(final_clip, style)
            if not enhanced_clip:
                enhanced_clip = final_clip
            
            # Final processing
            if enhanced_clip != output_path:
                final_processed = self._final_processing(enhanced_clip, output_path, style)
                if final_processed:
                    console.print(f"[green]Single viral clip created: {output_path}[/green]")
                    return output_path
            
            return enhanced_clip
            
        except Exception as e:
            console.print(f"[red]Error creating single long clip: {e}[/red]")
            return None
    
    def _select_moments_for_duration(self, viral_moments: List[ViralMoment], 
                                   target_duration: float) -> List[ViralMoment]:
        """Select and adjust viral moments to fit target duration."""
        
        # Sort by virality score (best first)
        sorted_moments = sorted(viral_moments, key=lambda m: m.virality_score, reverse=True)
        
        selected_moments = []
        total_duration = 0.0
        
        for moment in sorted_moments:
            moment_duration = moment.duration
            
            # If this moment would exceed target, adjust it
            if total_duration + moment_duration > target_duration:
                remaining_time = target_duration - total_duration
                if remaining_time >= 10.0:  # Minimum segment length
                    # Adjust moment duration
                    adjusted_moment = ViralMoment(
                        start_time=moment.start_time,
                        end_time=moment.start_time + remaining_time,
                        virality_score=moment.virality_score,
                        transcript_segments=moment.transcript_segments,
                        visual_highlights=moment.visual_highlights,
                        audio_features=moment.audio_features,
                        confidence=moment.confidence,
                        tags=moment.tags
                    )
                    selected_moments.append(adjusted_moment)
                    total_duration = target_duration
                break
            
            selected_moments.append(moment)
            total_duration += moment_duration
            
            # Stop if we've reached target duration
            if total_duration >= target_duration:
                break
        
        console.print(f"[blue]Selected {len(selected_moments)} moments for {total_duration:.1f}s total duration[/blue]")
        return selected_moments
    
    def _concatenate_clips_with_transitions(self, clip_paths: List[Path], 
                                          style: ClipStyle) -> Optional[Path]:
        """Concatenate clips with smooth transitions."""
        
        if len(clip_paths) == 1:
            return clip_paths[0]
        
        output_path = self.temp_dir / "concatenated_viral_clip.mp4"
        
        # Create input file list
        input_list = self.temp_dir / "clip_list.txt"
        
        try:
            with open(input_list, 'w') as f:
                for clip_path in clip_paths:
                    f.write(f"file '{clip_path.absolute()}'\n")
            
            # Concatenate with crossfade transitions
            if len(clip_paths) == 2:
                # Simple crossfade for 2 clips
                cmd = [
                    'ffmpeg', 
                    '-i', str(clip_paths[0]),
                    '-i', str(clip_paths[1]),
                    '-filter_complex', '[0][1]xfade=transition=fade:duration=1:offset=0',
                    '-y', str(output_path)
                ]
            else:
                # For multiple clips, use concat with fade transitions
                cmd = [
                    'ffmpeg', '-f', 'concat', '-safe', '0',
                    '-i', str(input_list),
                    '-vf', 'fade=t=in:st=0:d=0.5,fade=t=out:d=0.5',
                    '-c:a', 'aac',
                    '-y', str(output_path)
                ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            console.print(f"[green]Clips concatenated with transitions[/green]")
            return output_path
            
        except subprocess.CalledProcessError as e:
            console.print(f"[yellow]Transition concatenation failed, using simple concat: {e}[/yellow]")
            # Fallback to simple concatenation
            return self._simple_concatenate(clip_paths)
    
    def _simple_concatenate(self, clip_paths: List[Path]) -> Optional[Path]:
        """Simple concatenation without transitions as fallback."""
        
        output_path = self.temp_dir / "simple_concatenated.mp4"
        input_list = self.temp_dir / "simple_clip_list.txt"
        
        try:
            with open(input_list, 'w') as f:
                for clip_path in clip_paths:
                    f.write(f"file '{clip_path.absolute()}'\n")
            
            cmd = [
                'ffmpeg', '-f', 'concat', '-safe', '0',
                '-i', str(input_list),
                '-c', 'copy',
                '-y', str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return output_path
            
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Simple concatenation failed: {e}[/red]")
            return None

    def create_compilation(self, clip_paths: List[Path], 
                          output_path: Optional[Path] = None) -> Optional[Path]:
        """Create a compilation video from multiple clips."""
        
        if len(clip_paths) < 2:
            console.print("[yellow]Need at least 2 clips for compilation[/yellow]")
            return None
        
        if not output_path:
            output_path = self.output_dir / "viral_compilation.mp4"
        
        return self._simple_concatenate(clip_paths)
    
    def _create_viral_segments(self, viral_moments: List[ViralMoment], 
                              target_duration: float) -> List[Dict[str, Any]]:
        """Create optimized viral segments for TikTok-style editing."""
        
        segments = []
        total_duration = 0.0
        
        # Sort moments by virality score
        sorted_moments = sorted(viral_moments, key=lambda m: m.virality_score, reverse=True)
        
        for moment in sorted_moments:
            if total_duration >= target_duration:
                break
            
            # Break long moments into shorter viral segments
            moment_duration = moment.duration
            remaining_target = target_duration - total_duration
            
            if moment_duration <= config.VIRAL_SEGMENT_MAX_DURATION:
                # Moment is already optimal size
                segment_duration = min(moment_duration, remaining_target)
                segments.append({
                    'start_time': moment.start_time,
                    'end_time': moment.start_time + segment_duration,
                    'virality_score': moment.virality_score,
                    'type': 'highlight',
                    'transcript_segments': moment.transcript_segments,
                    'original_moment': moment
                })
                total_duration += segment_duration
            else:
                # Break long moment into multiple segments
                current_start = moment.start_time
                segments_from_moment = 0
                max_segments_per_moment = 3  # Limit segments per moment
                
                while (current_start < moment.end_time and 
                       total_duration < target_duration and 
                       segments_from_moment < max_segments_per_moment):
                    
                    remaining_in_moment = moment.end_time - current_start
                    remaining_target = target_duration - total_duration
                    
                    # Use target segment duration but adapt to available time
                    segment_duration = min(
                        config.VIRAL_SEGMENT_TARGET_DURATION,
                        remaining_in_moment,
                        remaining_target
                    )
                    
                    if segment_duration < config.VIRAL_SEGMENT_MIN_DURATION:
                        break
                    
                    # Find relevant transcript for this segment
                    segment_transcript = [
                        seg for seg in moment.transcript_segments 
                        if seg.start >= current_start and seg.end <= current_start + segment_duration
                    ]
                    
                    segments.append({
                        'start_time': current_start,
                        'end_time': current_start + segment_duration,
                        'virality_score': moment.virality_score * (1 - segments_from_moment * 0.1),  # Slight decrease for later segments
                        'type': 'rapid_cut',
                        'transcript_segments': segment_transcript,
                        'original_moment': moment
                    })
                    
                    current_start += segment_duration
                    total_duration += segment_duration
                    segments_from_moment += 1
            
            # Stop if we have enough segments
            if len(segments) >= config.MAX_VIRAL_SEGMENTS:
                break
        
        # Sort segments by start time for chronological flow
        segments.sort(key=lambda s: s['start_time'])
        
        console.print(f"[blue]Created {len(segments)} viral segments totaling {total_duration:.1f}s[/blue]")
        return segments
    
    def _extract_viral_segment(self, video_path: Path, segment: Dict[str, Any], 
                              style: ClipStyle, index: int) -> Optional[Path]:
        """Extract a single viral segment with optimizations."""
        
        start_time = segment['start_time']
        end_time = segment['end_time']
        duration = end_time - start_time
        
        output_path = self.temp_dir / f"viral_segment_{index:03d}_{start_time:.1f}s.mp4"
        
        # Minimal padding to avoid issues
        padding = 0.05
        padded_start = max(0, start_time - padding)
        padded_duration = duration + (2 * padding)
        
        # Simplified command without complex scaling
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-ss', str(padded_start),
            '-t', str(padded_duration),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-preset', 'fast',
            '-crf', '23',
            '-avoid_negative_ts', 'make_zero',  # Avoid timing issues
            '-y', str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Verify the output file exists and has content
            if output_path.exists() and output_path.stat().st_size > 1024:  # At least 1KB
                console.print(f"[green]Extracted segment {index}: {duration:.1f}s[/green]")
                return output_path
            else:
                console.print(f"[yellow]Segment {index} file is empty or too small[/yellow]")
                return None
                
        except subprocess.CalledProcessError as e:
            console.print(f"[yellow]Failed to extract viral segment {index}: {e.stderr if e.stderr else str(e)}[/yellow]")
            return None
    
    def _create_viral_compilation(self, segment_clips: List[Path], style: ClipStyle) -> Optional[Path]:
        """Create viral compilation with rapid cuts and transitions."""
        
        output_path = self.temp_dir / "viral_compilation.mp4"
        
        if len(segment_clips) == 1:
            console.print("[blue]Single segment, copying directly[/blue]")
            import shutil
            shutil.copy2(segment_clips[0], output_path)
            return output_path
        
        # Verify all segments exist and have content
        valid_clips = []
        for clip in segment_clips:
            if clip.exists() and clip.stat().st_size > 1024:
                valid_clips.append(clip)
            else:
                console.print(f"[yellow]Skipping invalid segment: {clip}[/yellow]")
        
        if not valid_clips:
            console.print("[red]No valid segments to concatenate[/red]")
            return None
        
        if len(valid_clips) == 1:
            import shutil
            shutil.copy2(valid_clips[0], output_path)
            return output_path
        
        # Create concat list
        input_list = self.temp_dir / "viral_segments.txt"
        
        try:
            with open(input_list, 'w') as f:
                for clip_path in valid_clips:
                    f.write(f"file '{clip_path.absolute()}'\n")
            
            # Simplified concatenation without complex filters
            cmd = [
                'ffmpeg', '-f', 'concat', '-safe', '0',
                '-i', str(input_list),
                '-c', 'copy',  # Copy streams without re-encoding
                '-avoid_negative_ts', 'make_zero',
                '-y', str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Verify output
            if output_path.exists() and output_path.stat().st_size > 1024:
                console.print(f"[green]Viral compilation created with {len(valid_clips)} segments[/green]")
                return output_path
            else:
                console.print("[yellow]Compilation file is empty, trying fallback[/yellow]")
                return self._simple_concatenate(valid_clips)
            
        except subprocess.CalledProcessError as e:
            console.print(f"[yellow]Viral compilation failed: {e.stderr if e.stderr else str(e)}[/yellow]")
            return self._simple_concatenate(valid_clips)
    
    def _apply_viral_effects(self, video_path: Path, style: ClipStyle, output_path: Path) -> Optional[Path]:
        """Apply viral TikTok-style effects and optimizations."""
        
        # Verify input file
        if not video_path.exists() or video_path.stat().st_size < 1024:
            console.print(f"[red]Input video invalid: {video_path}[/red]")
            return None
        
        # Simple processing without complex filters to avoid black screen
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-c:v', 'libx264',
            '-preset', 'fast',  # Faster processing
            '-crf', '23',  # Good quality
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            '-avoid_negative_ts', 'make_zero',
            '-y', str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Verify output
            if output_path.exists() and output_path.stat().st_size > 1024:
                console.print("[green]Viral clip finalized successfully[/green]")
                return output_path
            else:
                console.print("[yellow]Final output is empty, copying input[/yellow]")
                import shutil
                shutil.copy2(video_path, output_path)
                return output_path
                
        except subprocess.CalledProcessError as e:
            console.print(f"[yellow]Final processing failed: {e.stderr if e.stderr else str(e)}[/yellow]")
            # Fallback to simple copy
            try:
                import shutil
                shutil.copy2(video_path, output_path)
                console.print("[blue]Using original compilation as fallback[/blue]")
                return output_path
            except Exception as copy_error:
                console.print(f"[red]Copy fallback failed: {copy_error}[/red]")
                return None
    
    def cleanup_temp_files(self):
        """Clean up temporary files created during processing."""
        self.video_processor.cleanup_temp_files()
        
        # Clean up our temp files
        for file in self.temp_dir.glob("temp_*"):
            try:
                file.unlink()
            except Exception as e:
                console.print(f"[yellow]Could not delete {file}: {e}[/yellow]")

def create_viral_clip(video_path: Path, viral_moment: ViralMoment, 
                     output_dir: Optional[Path] = None) -> Optional[Path]:
    """Convenience function to create a viral clip."""
    assembler = ClipAssembler(output_dir)
    return assembler.create_viral_clip(video_path, viral_moment)