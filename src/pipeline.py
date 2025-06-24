"""Main pipeline for viral clip generation."""

import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table
from rich.panel import Panel

from .config import config
from .downloader import VideoDownloader
from .video_processor import VideoProcessor
from .speech_to_text import SpeechToText, TranscriptSegment
from .visual_analysis import VisualAnalyzer
from .virality_scorer import ViralityScorer, ViralMoment
from .clip_assembler import ClipAssembler, ClipStyle

console = Console()

@dataclass
class PipelineResult:
    """Results from the viral clip generation pipeline."""
    success: bool
    video_info: Dict[str, Any]
    transcript_segments: List[TranscriptSegment]
    visual_analysis: Dict[str, Any]
    viral_moments: List[ViralMoment]
    generated_clips: List[Path]
    thumbnails: List[Path]
    processing_time: float
    error_message: Optional[str] = None

class ViralClipPipeline:
    """Complete pipeline for generating viral clips from YouTube videos."""
    
    def __init__(self, temp_dir: Optional[Path] = None, output_dir: Optional[Path] = None):
        self.temp_dir = temp_dir or config.TEMP_DIR
        self.output_dir = output_dir or config.OUTPUTS_DIR
        
        # Initialize components
        self.downloader = VideoDownloader(self.temp_dir)
        self.video_processor = VideoProcessor(self.temp_dir)
        self.speech_to_text = SpeechToText()
        self.visual_analyzer = VisualAnalyzer()
        self.virality_scorer = ViralityScorer()
        self.clip_assembler = ClipAssembler(self.output_dir, self.temp_dir)
        
        console.print("[green]✅ Viral Clip Pipeline initialized[/green]")
    
    def process_video(self, url_or_path: str, 
                     max_clips: int = 3,
                     min_clip_duration: float = 15.0,
                     max_clip_duration: float = 60.0,
                     clip_style: Optional[ClipStyle] = None) -> PipelineResult:
        """
        Process a video through the complete viral clip generation pipeline.
        
        Args:
            url_or_path: YouTube URL or local video file path
            max_clips: Maximum number of clips to generate
            min_clip_duration: Minimum clip duration in seconds
            max_clip_duration: Maximum clip duration in seconds
            clip_style: Custom clip styling options
        """
        
        start_time = time.time()
        
        console.print(Panel.fit(
            f"🎬 [bold blue]VIRAL CLIP GENERATOR[/bold blue] 🎬\n"
            f"Processing: {url_or_path}",
            style="blue"
        ))
        
        with Progress() as progress:
            # Create progress tasks
            overall_task = progress.add_task("[cyan]Overall Progress", total=100)
            current_task = progress.add_task("[green]Current Step", total=100)
            
            try:
                # Step 1: Download/Load Video (15%)
                progress.update(current_task, description="[green]Downloading video...", completed=0)
                video_path = self._download_or_load_video(url_or_path, progress, current_task)
                if not video_path:
                    return self._create_error_result("Failed to download/load video", time.time() - start_time)
                
                progress.update(overall_task, completed=15)
                
                # Step 2: Extract Audio and Video Info (25%)
                progress.update(current_task, description="[green]Processing video...", completed=0)
                audio_path, video_info = self._process_video(video_path, progress, current_task)
                if not audio_path:
                    return self._create_error_result("Failed to process video", time.time() - start_time)
                
                progress.update(overall_task, completed=25)
                
                # Step 3: Speech-to-Text (40%)
                progress.update(current_task, description="[green]Transcribing audio...", completed=0)
                transcript_segments = self._transcribe_audio(audio_path, progress, current_task)
                
                progress.update(overall_task, completed=40)
                
                # Step 4: Visual Analysis (60%)
                progress.update(current_task, description="[green]Analyzing visuals...", completed=0)
                visual_analysis = self._analyze_visuals(video_path, progress, current_task)
                
                progress.update(overall_task, completed=60)
                
                # Step 5: Find Viral Moments (75%)
                progress.update(current_task, description="[green]Finding viral moments...", completed=0)
                viral_moments = self._find_viral_moments(
                    transcript_segments, visual_analysis, 
                    min_clip_duration, max_clip_duration,
                    progress, current_task
                )
                
                progress.update(overall_task, completed=75)
                
                # Step 6: Generate TikTok Viral Clip (100%)
                progress.update(current_task, description="[green]Creating TikTok viral clip...", completed=0)
                clips, thumbnails = self._generate_tiktok_viral_clip(
                    video_path, viral_moments, clip_style,
                    progress, current_task
                )
                
                progress.update(overall_task, completed=100)
                progress.update(current_task, completed=100)
                
            except Exception as e:
                processing_time = time.time() - start_time
                console.print(f"[red]Error during processing: {str(e)}[/red]")
                return self._create_error_result(str(e), processing_time)
        
        processing_time = time.time() - start_time
        
        # Display results
        self._display_results(viral_moments, clips, thumbnails, processing_time)
        
        # Cleanup
        self._cleanup_temp_files()
        
        return PipelineResult(
            success=True,
            video_info=video_info,
            transcript_segments=transcript_segments,
            visual_analysis=visual_analysis,
            viral_moments=viral_moments,
            generated_clips=clips,
            thumbnails=thumbnails,
            processing_time=processing_time
        )
    
    def _download_or_load_video(self, url_or_path: str, progress: Progress, task: TaskID) -> Optional[Path]:
        """Download video from URL or load local file."""
        
        if url_or_path.startswith(('http://', 'https://')):
            # Download from URL
            progress.update(task, description="[green]Downloading from YouTube...", completed=20)
            
            # Get video info first
            video_info = self.downloader.get_video_info(url_or_path)
            if not video_info:
                console.print("[red]❌ Failed to get video information[/red]")
                return None
            
            console.print(f"[blue]📺 Title: {video_info.get('title', 'Unknown')}[/blue]")
            console.print(f"[blue]⏱️  Duration: {video_info.get('duration', 0)/60:.1f} minutes[/blue]")
            
            progress.update(task, completed=50)
            
            # Download video
            video_path = self.downloader.download_video(url_or_path)
            progress.update(task, completed=100)
            
            return video_path
        else:
            # Load local file
            video_path = Path(url_or_path)
            if not video_path.exists():
                console.print(f"[red]❌ Video file not found: {video_path}[/red]")
                return None
            
            progress.update(task, completed=100)
            console.print(f"[blue]📁 Loaded local video: {video_path.name}[/blue]")
            return video_path
    
    def _process_video(self, video_path: Path, progress: Progress, task: TaskID) -> tuple[Optional[Path], Dict[str, Any]]:
        """Process video to extract audio and get video information."""
        
        # Get video info
        progress.update(task, completed=25)
        video_info = self.video_processor.get_video_info(video_path)
        
        if not video_info:
            console.print("[red]❌ Failed to get video information[/red]")
            return None, {}
        
        # Extract audio
        progress.update(task, completed=50)
        audio_path = self.video_processor.extract_audio(video_path)
        progress.update(task, completed=100)
        
        if not audio_path:
            console.print("[red]❌ Failed to extract audio[/red]")
            return None, video_info
        
        console.print(f"[blue]🎵 Audio extracted: {audio_path.name}[/blue]")
        return audio_path, video_info
    
    def _transcribe_audio(self, audio_path: Path, progress: Progress, task: TaskID) -> List[TranscriptSegment]:
        """Transcribe audio to text."""
        
        progress.update(task, completed=20)
        
        try:
            segments = self.speech_to_text.transcribe_audio(audio_path)
            progress.update(task, completed=80)
            
            # Enhance with speaker detection
            enhanced_segments = self.speech_to_text.detect_speaker_changes(segments)
            progress.update(task, completed=100)
            
            console.print(f"[blue]📝 Transcribed {len(enhanced_segments)} segments[/blue]")
            return enhanced_segments
            
        except Exception as e:
            console.print(f"[yellow]⚠️  Transcription failed: {e}[/yellow]")
            return []
    
    def _analyze_visuals(self, video_path: Path, progress: Progress, task: TaskID) -> Dict[str, Any]:
        """Analyze video for visual features."""
        
        progress.update(task, completed=10)
        
        try:
            # Extract frames for analysis
            frames = self.video_processor.extract_frames(video_path, fps=0.5)  # 0.5 fps for analysis
            progress.update(task, completed=40)
            
            if not frames:
                console.print("[yellow]⚠️  No frames extracted for visual analysis[/yellow]")
                return {}
            
            # Analyze frames
            analysis = self.visual_analyzer.analyze_frames(frames, fps=0.5)
            progress.update(task, completed=90)
            
            # Find viral visual moments
            viral_visual_moments = self.visual_analyzer.detect_viral_moments(analysis)
            analysis['viral_visual_moments'] = viral_visual_moments
            
            progress.update(task, completed=100)
            
            console.print(f"[blue]👁️  Visual analysis: {len(analysis.get('objects', []))} objects, "
                         f"{len(analysis.get('emotions', []))} emotions detected[/blue]")
            
            return analysis
            
        except Exception as e:
            console.print(f"[yellow]⚠️  Visual analysis failed: {e}[/yellow]")
            return {}
    
    def _find_viral_moments(self, transcript_segments: List[TranscriptSegment],
                           visual_analysis: Dict[str, Any],
                           min_duration: float, max_duration: float,
                           progress: Progress, task: TaskID) -> List[ViralMoment]:
        """Find viral moments in the video."""
        
        progress.update(task, completed=20)
        
        try:
            # Extract visual features
            objects = visual_analysis.get('objects', [])
            emotions = visual_analysis.get('emotions', [])
            actions = visual_analysis.get('actions', [])
            
            progress.update(task, completed=50)
            
            # Find viral moments
            viral_moments = self.virality_scorer.find_viral_moments(
                transcript_segments=transcript_segments,
                visual_objects=objects,
                emotions=emotions,
                actions=actions,
                silence_segments=[],
                min_duration=min_duration,
                max_duration=max_duration
            )
            
            progress.update(task, completed=100)
            
            console.print(f"[blue]🔥 Found {len(viral_moments)} viral moments[/blue]")
            
            # Display top moments
            if viral_moments:
                self._display_viral_moments_preview(viral_moments[:3])
            
            return viral_moments
            
        except Exception as e:
            console.print(f"[yellow]⚠️  Viral moment detection failed: {e}[/yellow]")
            return []
    
    def _generate_single_long_clip(self, video_path: Path, viral_moments: List[ViralMoment],
                                  clip_style: Optional[ClipStyle],
                                  progress: Progress, task: TaskID) -> tuple[List[Path], List[Path]]:
        """Generate a single long clip from viral moments."""
        
        if not viral_moments:
            console.print("[yellow]⚠️  No viral moments to generate clip from[/yellow]")
            return [], []
        
        clips = []
        thumbnails = []
        
        try:
            progress.update(task, description="[green]Creating single viral highlight...", completed=20)
            
            # Create single long clip using target duration from config
            target_duration = config.TARGET_CLIP_DURATION
            clip_path = self.clip_assembler.create_single_long_clip(
                video_path, viral_moments, target_duration, clip_style
            )
            
            progress.update(task, completed=80)
            
            if clip_path:
                clips.append(clip_path)
                
                # Create thumbnail from the best moment
                best_moment = max(viral_moments, key=lambda m: m.virality_score)
                thumbnail_path = self.clip_assembler.create_thumbnail(
                    video_path, best_moment
                )
                
                if thumbnail_path:
                    thumbnails.append(thumbnail_path)
                
                progress.update(task, completed=100)
                console.print(f"[blue]🎬 Generated single {target_duration}s viral highlight clip[/blue]")
            else:
                console.print("[yellow]⚠️  Failed to create viral highlight clip[/yellow]")
                
        except Exception as e:
            console.print(f"[yellow]⚠️  Failed to create viral highlight: {e}[/yellow]")
        
        return clips, thumbnails

    def _generate_tiktok_viral_clip(self, video_path: Path, viral_moments: List[ViralMoment],
                                   clip_style: Optional[ClipStyle],
                                   progress: Progress, task: TaskID) -> tuple[List[Path], List[Path]]:
        """Generate a TikTok-style viral clip with rapid cuts and high engagement."""
        
        if not viral_moments:
            console.print("[yellow]⚠️  No viral moments to generate clip from[/yellow]")
            return [], []
        
        clips = []
        thumbnails = []
        
        try:
            progress.update(task, description="[green]Creating TikTok viral clip...", completed=20)
            
            # Create TikTok-style viral clip with rapid cuts
            target_duration = config.TARGET_CLIP_DURATION
            clip_path = self.clip_assembler.create_tiktok_viral_clip(
                video_path, viral_moments, target_duration, clip_style
            )
            
            progress.update(task, completed=80)
            
            if clip_path:
                clips.append(clip_path)
                
                # Create viral thumbnail from the best moment
                best_moment = max(viral_moments, key=lambda m: m.virality_score)
                thumbnail_path = self.clip_assembler.create_thumbnail(
                    video_path, best_moment
                )
                
                if thumbnail_path:
                    thumbnails.append(thumbnail_path)
                
                progress.update(task, completed=100)
                console.print(f"[blue]🎬 Generated TikTok viral clip with rapid cuts[/blue]")
            else:
                console.print("[yellow]⚠️  Failed to create TikTok viral clip[/yellow]")
                
        except Exception as e:
            console.print(f"[yellow]⚠️  Failed to create TikTok viral clip: {e}[/yellow]")
        
        return clips, thumbnails

    def _generate_clips(self, video_path: Path, viral_moments: List[ViralMoment],
                       clip_style: Optional[ClipStyle],
                       progress: Progress, task: TaskID) -> tuple[List[Path], List[Path]]:
        """Generate viral clips from viral moments (legacy method)."""
        
        if not viral_moments:
            console.print("[yellow]⚠️  No viral moments to generate clips from[/yellow]")
            return [], []
        
        clips = []
        thumbnails = []
        
        total_moments = len(viral_moments)
        
        for i, moment in enumerate(viral_moments):
            progress.update(task, 
                          description=f"[green]Creating clip {i+1}/{total_moments}...",
                          completed=int((i / total_moments) * 80))
            
            try:
                # Create clip
                clip_path = self.clip_assembler.create_viral_clip(
                    video_path, moment, clip_style
                )
                
                if clip_path:
                    clips.append(clip_path)
                    
                    # Create thumbnail
                    thumbnail_path = self.clip_assembler.create_thumbnail(
                        video_path, moment
                    )
                    
                    if thumbnail_path:
                        thumbnails.append(thumbnail_path)
                
            except Exception as e:
                console.print(f"[yellow]⚠️  Failed to create clip {i+1}: {e}[/yellow]")
                continue
        
        progress.update(task, completed=100)
        
        console.print(f"[blue]🎬 Generated {len(clips)} viral clips[/blue]")
        return clips, thumbnails
    
    def _display_viral_moments_preview(self, moments: List[ViralMoment]):
        """Display a preview of the top viral moments."""
        
        table = Table(title="🔥 Top Viral Moments", show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Time", style="cyan", width=12)
        table.add_column("Duration", style="green", width=10)
        table.add_column("Score", style="red", width=8)
        table.add_column("Preview", style="white", width=50)
        
        for i, moment in enumerate(moments, 1):
            time_str = f"{moment.start_time:.1f}s"
            duration_str = f"{moment.duration:.1f}s"
            score_str = f"{moment.virality_score:.2f}"
            
            # Create preview from transcript
            preview = ""
            if moment.transcript_segments:
                all_text = " ".join(seg.text for seg in moment.transcript_segments)
                preview = (all_text[:47] + "...") if len(all_text) > 50 else all_text
            
            table.add_row(
                str(i),
                time_str,
                duration_str,
                score_str,
                preview
            )
        
        console.print(table)
    
    def _display_results(self, viral_moments: List[ViralMoment], 
                        clips: List[Path], thumbnails: List[Path],
                        processing_time: float):
        """Display final results."""
        
        results_table = Table(title="📊 Processing Results", show_header=True)
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")
        
        results_table.add_row("Viral Moments Found", str(len(viral_moments)))
        results_table.add_row("Clips Generated", str(len(clips)))
        results_table.add_row("Thumbnails Created", str(len(thumbnails)))
        results_table.add_row("Processing Time", f"{processing_time:.1f}s")
        
        console.print(results_table)
        
        if clips:
            console.print("\n[green]✅ Generated Clips:[/green]")
            for i, clip_path in enumerate(clips, 1):
                console.print(f"  {i}. {clip_path}")
        
        if thumbnails:
            console.print("\n[green]🖼️  Generated Thumbnails:[/green]")
            for i, thumb_path in enumerate(thumbnails, 1):
                console.print(f"  {i}. {thumb_path}")
    
    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        try:
            self.downloader.cleanup_temp_files()
            self.video_processor.cleanup_temp_files()
            self.clip_assembler.cleanup_temp_files()
        except Exception as e:
            console.print(f"[yellow]⚠️  Cleanup warning: {e}[/yellow]")
    
    def _create_error_result(self, error_message: str, processing_time: float) -> PipelineResult:
        """Create an error result."""
        console.print(f"[red]❌ {error_message}[/red]")
        
        return PipelineResult(
            success=False,
            video_info={},
            transcript_segments=[],
            visual_analysis={},
            viral_moments=[],
            generated_clips=[],
            thumbnails=[],
            processing_time=processing_time,
            error_message=error_message
        )
    
    def batch_process(self, urls_or_paths: List[str], **kwargs) -> List[PipelineResult]:
        """Process multiple videos in batch."""
        
        results = []
        
        console.print(f"[blue]🎬 Starting batch processing of {len(urls_or_paths)} videos[/blue]")
        
        for i, url_or_path in enumerate(urls_or_paths, 1):
            console.print(f"\n[cyan]Processing video {i}/{len(urls_or_paths)}[/cyan]")
            
            try:
                result = self.process_video(url_or_path, **kwargs)
                results.append(result)
                
                if result.success:
                    console.print(f"[green]✅ Video {i} processed successfully[/green]")
                else:
                    console.print(f"[red]❌ Video {i} failed: {result.error_message}[/red]")
                    
            except Exception as e:
                console.print(f"[red]❌ Video {i} crashed: {e}[/red]")
                results.append(self._create_error_result(str(e), 0.0))
        
        # Summary
        successful = sum(1 for r in results if r.success)
        console.print(f"\n[blue]📊 Batch Results: {successful}/{len(results)} successful[/blue]")
        
        return results

def process_video(url_or_path: str, **kwargs) -> PipelineResult:
    """Convenience function to process a single video."""
    pipeline = ViralClipPipeline()
    return pipeline.process_video(url_or_path, **kwargs)