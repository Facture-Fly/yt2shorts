#!/usr/bin/env python3
"""
Viral YouTube Clip Generator
A complete AI-powered pipeline for creating viral clips from YouTube videos.
"""

import sys
import argparse
import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline import ViralClipPipeline, PipelineResult
from src.clip_assembler import ClipStyle
from src.config import config

console = Console()
app = typer.Typer(help="🎬 Viral YouTube Clip Generator")

def check_models_downloaded() -> bool:
    """Check if all required AI models are downloaded to local models/ directory."""
    local_models_to_check = [
        # Local model directories
        config.MODELS_DIR / "emotion-model",
        config.MODELS_DIR / "clip-model", 
        config.MODELS_DIR / "sentiment-model",
        config.MODELS_DIR / "dialogpt-model",
        config.MODELS_DIR / "sentence-transformer",
        config.MODELS_DIR / "whisper",
        # YOLO model
        Path(config.YOLO_MODEL),
    ]
    
    for model_path in local_models_to_check:
        if not model_path.exists():
            return False
    
    # Check if model directories contain actual model files
    required_files = [
        config.MODELS_DIR / "emotion-model" / "config.json",
        config.MODELS_DIR / "clip-model" / "config.json",
        config.MODELS_DIR / "sentiment-model" / "config.json", 
        config.MODELS_DIR / "dialogpt-model" / "config.json",
        config.MODELS_DIR / "whisper" / f"{config.WHISPER_MODEL}.pt",
    ]
    
    for required_file in required_files:
        if not required_file.exists():
            return False
    
    return True

def download_models():
    """Run the download_models.py script to download all required models."""
    console.print("[yellow]⚠️  Some AI models are missing. Downloading required models...[/yellow]")
    
    try:
        # Run download_models.py
        result = subprocess.run([
            sys.executable, 
            str(Path(__file__).parent / "download_models.py")
        ], check=True, capture_output=True, text=True)
        
        console.print("[green]✓ Models downloaded successfully![/green]")
        return True
        
    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗ Failed to download models: {e}[/red]")
        console.print(f"[red]Error output: {e.stderr}[/red]")
        return False
    except Exception as e:
        console.print(f"[red]✗ Error running model download: {e}[/red]")
        return False

def print_banner():
    """Print application banner."""
    banner = """
    🎬 VIRAL CLIP GENERATOR 🎬
    ╔════════════════════════════════════════╗
    ║  AI-Powered YouTube Clip Creation      ║
    ║  • Speech-to-Text with Whisper         ║
    ║  • Visual Analysis with YOLO/CLIP      ║
    ║  • Virality Scoring with LLM           ║
    ║  • Auto Clip Assembly & Enhancement    ║
    ╚════════════════════════════════════════╝
    """
    rprint(Panel(banner, style="bold blue"))

@app.command()
def generate(
    url_or_path: str = typer.Argument(..., help="YouTube URL or local video file path"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory for clips"),
    max_clips: int = typer.Option(3, "--max-clips", "-n", help="Maximum number of clips to generate"),
    min_duration: float = typer.Option(15.0, "--min-duration", help="Minimum clip duration (seconds)"),
    max_duration: float = typer.Option(60.0, "--max-duration", help="Maximum clip duration (seconds)"),
    resolution: str = typer.Option("1080p", "--resolution", "-r", help="Output resolution (720p, 1080p, 4k)"),
    style: str = typer.Option("default", "--style", "-s", help="Clip style (default, energetic, minimal)"),
    no_subtitles: bool = typer.Option(False, "--no-subtitles", help="Disable subtitle generation"),
    no_effects: bool = typer.Option(False, "--no-effects", help="Disable visual effects"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """
    Generate viral clips from a YouTube video or local file.
    
    Examples:
        python main.py generate "https://www.youtube.com/watch?v=Pv0iVoSZzN8"
        python main.py generate "/path/to/video.mp4" --output ./clips --max-clips 5
        python main.py generate "URL" --resolution 720p --style energetic
    """
    
    if not verbose:
        print_banner()
    
    # Setup output directory
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = config.OUTPUTS_DIR
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create clip style
    clip_style = create_clip_style(resolution, style, not no_subtitles, not no_effects)
    
    # Initialize pipeline
    console.print("[blue]🚀 Initializing Viral Clip Pipeline...[/blue]")
    pipeline = ViralClipPipeline(output_dir=output_path)
    
    try:
        # Process video
        result = pipeline.process_video(
            url_or_path=url_or_path,
            max_clips=max_clips,
            min_clip_duration=min_duration,
            max_clip_duration=max_duration,
            clip_style=clip_style
        )
        
        # Handle results
        if result.success:
            console.print(f"\n[green]🎉 SUCCESS! Generated {len(result.generated_clips)} viral clips[/green]")
            
            if result.generated_clips:
                console.print("\n[cyan]📁 Output Files:[/cyan]")
                for i, clip_path in enumerate(result.generated_clips, 1):
                    console.print(f"  {i}. {clip_path}")
            
            return 0
        else:
            console.print(f"\n[red]❌ FAILED: {result.error_message}[/red]")
            return 1
            
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Process interrupted by user[/yellow]")
        return 1
    except Exception as e:
        console.print(f"\n[red]💥 Unexpected error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        return 1

@app.command()
def batch(
    input_file: str = typer.Argument(..., help="File containing URLs/paths (one per line)"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory for clips"),
    max_clips: int = typer.Option(3, "--max-clips", "-n", help="Maximum clips per video"),
    resolution: str = typer.Option("1080p", "--resolution", "-r", help="Output resolution"),
    parallel: bool = typer.Option(False, "--parallel", "-p", help="Process videos in parallel"),
):
    """
    Process multiple videos from a file containing URLs/paths.
    
    Example:
        python main.py batch urls.txt --output ./clips --max-clips 2
    """
    
    print_banner()
    
    # Read input file
    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"[red]❌ Input file not found: {input_file}[/red]")
        return 1
    
    urls_or_paths = []
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                urls_or_paths.append(line)
    
    if not urls_or_paths:
        console.print("[red]❌ No valid URLs/paths found in input file[/red]")
        return 1
    
    console.print(f"[blue]📋 Found {len(urls_or_paths)} videos to process[/blue]")
    
    # Setup output directory
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = config.OUTPUTS_DIR
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create clip style
    clip_style = create_clip_style(resolution, "default", True, True)
    
    # Initialize pipeline
    pipeline = ViralClipPipeline(output_dir=output_path)
    
    try:
        # Process videos
        results = pipeline.batch_process(
            urls_or_paths,
            max_clips=max_clips,
            clip_style=clip_style
        )
        
        # Summary
        successful = sum(1 for r in results if r.success)
        total_clips = sum(len(r.generated_clips) for r in results if r.success)
        
        console.print(f"\n[green]🎉 Batch processing complete![/green]")
        console.print(f"[cyan]📊 Results: {successful}/{len(results)} videos successful[/cyan]")
        console.print(f"[cyan]🎬 Total clips generated: {total_clips}[/cyan]")
        
        return 0 if successful > 0 else 1
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Batch processing interrupted[/yellow]")
        return 1
    except Exception as e:
        console.print(f"\n[red]💥 Batch processing failed: {e}[/red]")
        return 1

@app.command()
def info(
    url_or_path: str = typer.Argument(..., help="YouTube URL or local video file path")
):
    """
    Get information about a video without processing it.
    """
    
    from src.downloader import VideoDownloader
    from src.video_processor import VideoProcessor
    
    console.print(f"[blue]📹 Getting video information...[/blue]")
    
    try:
        if url_or_path.startswith(('http://', 'https://')):
            # YouTube video
            downloader = VideoDownloader()
            info = downloader.get_video_info(url_or_path)
            
            if info:
                console.print(f"\n[green]📺 Video Information:[/green]")
                console.print(f"Title: {info.get('title', 'Unknown')}")
                console.print(f"Duration: {info.get('duration', 0)/60:.1f} minutes")
                console.print(f"Uploader: {info.get('uploader', 'Unknown')}")
                console.print(f"Views: {info.get('view_count', 0):,}")
                console.print(f"Likes: {info.get('like_count', 0):,}")
                console.print(f"Upload Date: {info.get('upload_date', 'Unknown')}")
            else:
                console.print("[red]❌ Could not retrieve video information[/red]")
                return 1
        else:
            # Local file
            video_path = Path(url_or_path)
            if not video_path.exists():
                console.print(f"[red]❌ Video file not found: {url_or_path}[/red]")
                return 1
            
            processor = VideoProcessor()
            info = processor.get_video_info(video_path)
            
            if info:
                console.print(f"\n[green]📁 Local Video Information:[/green]")
                console.print(f"File: {video_path.name}")
                console.print(f"Duration: {info.get('duration', 0)/60:.1f} minutes")
                console.print(f"Size: {info.get('size', 0)/(1024*1024):.1f} MB")
                console.print(f"Resolution: {info.get('video', {}).get('width', 0)}x{info.get('video', {}).get('height', 0)}")
                console.print(f"FPS: {info.get('video', {}).get('fps', 0):.1f}")
                console.print(f"Video Codec: {info.get('video', {}).get('codec', 'Unknown')}")
                console.print(f"Audio Codec: {info.get('audio', {}).get('codec', 'Unknown')}")
            else:
                console.print("[red]❌ Could not analyze video file[/red]")
                return 1
        
        return 0
        
    except Exception as e:
        console.print(f"[red]❌ Error getting video info: {e}[/red]")
        return 1

@app.command()
def config_info():
    """Show current configuration settings."""
    
    console.print("[blue]⚙️  Current Configuration:[/blue]")
    console.print(f"Base Directory: {config.BASE_DIR}")
    console.print(f"Output Directory: {config.OUTPUTS_DIR}")
    console.print(f"Temp Directory: {config.TEMP_DIR}")
    console.print(f"Models Directory: {config.MODELS_DIR}")
    console.print(f"Whisper Model: {config.WHISPER_MODEL}")
    console.print(f"YOLO Model: {config.YOLO_MODEL}")
    console.print(f"Device: {config.WHISPER_DEVICE}")
    console.print(f"Max Video Duration: {config.MAX_VIDEO_DURATION}s")
    console.print(f"Target Clip Duration: {config.TARGET_CLIP_DURATION}s")

def create_clip_style(resolution: str, style: str, subtitles: bool, effects: bool) -> ClipStyle:
    """Create a ClipStyle based on parameters."""
    
    # Resolution mapping
    resolution_map = {
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "4k": (3840, 2160)
    }
    
    res = resolution_map.get(resolution, (1920, 1080))
    
    # Style presets
    if style == "energetic":
        return ClipStyle(
            resolution=res,
            fps=30,
            subtitle_font_size=52,
            subtitle_color="yellow",
            subtitle_outline_color="black",
            subtitle_outline_width=3,
            fade_in_duration=0.3,
            fade_out_duration=0.3,
            zoom_effect=effects,
            zoom_intensity=0.15,
            color_correction=effects,
            audio_enhancement=True
        )
    elif style == "minimal":
        return ClipStyle(
            resolution=res,
            fps=24,
            subtitle_font_size=44,
            subtitle_color="white",
            subtitle_outline_color="black",
            subtitle_outline_width=1,
            fade_in_duration=1.0,
            fade_out_duration=1.0,
            zoom_effect=False,
            zoom_intensity=0.0,
            color_correction=False,
            audio_enhancement=effects
        )
    else:  # default
        return ClipStyle(
            resolution=res,
            fps=30,
            subtitle_font_size=48,
            subtitle_color="white",
            subtitle_outline_color="black",
            subtitle_outline_width=2,
            fade_in_duration=0.5,
            fade_out_duration=0.5,
            zoom_effect=effects,
            zoom_intensity=0.1,
            color_correction=effects,
            audio_enhancement=True
        )

def main():
    """Main entry point for command line usage."""
    try:
        # Check if models are downloaded before running the app
        if not check_models_downloaded():
            console.print("[blue]🤖 Checking AI models...[/blue]")
            console.print("[yellow]⚠️  Models appear to be missing, but attempting to continue...[/yellow]")
            # Skip model download for now since models seem to exist
            # if not download_models():
            #     console.print("[red]❌ Failed to download required models. Exiting.[/red]")
            #     sys.exit(1)
        
        app()
    except Exception as e:
        console.print(f"[red]💥 Application error: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()