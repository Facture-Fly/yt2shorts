#!/usr/bin/env python3
"""
TikTok/YouTube Shorts Style Video Editor
Creates mobile-optimized viral videos with captions, zooms, speed changes, and trending effects.
"""

import typer
import sys
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent))

from video_edition.shorts_editor import ShortsVideoEditor
from video_edition.config import ShortsConfig

console = Console()

def create_shorts(
    path: str = typer.Argument(..., help="Path to input video file"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    duration: float = typer.Option(60.0, "--duration", "-d", help="Target duration in seconds"),
    aspect_ratio: str = typer.Option("9:16", "--aspect-ratio", "-ar", help="Aspect ratio (9:16, 16:9, 1:1)"),
    add_captions: bool = typer.Option(True, "--captions/--no-captions", help="Add automatic captions"),
    add_zooms: bool = typer.Option(True, "--zooms/--no-zooms", help="Add dynamic zoom effects"),
    speed_variations: bool = typer.Option(True, "--speed/--no-speed", help="Add speed variations"),
    add_music: bool = typer.Option(False, "--music/--no-music", help="Add trending background music"),
    text_style: str = typer.Option("bold", "--text-style", help="Caption style: bold, minimal, colorful"),
    zoom_intensity: float = typer.Option(1.2, "--zoom-intensity", help="Zoom effect intensity (1.0-2.0)"),
    speed_factor: float = typer.Option(1.5, "--speed-factor", help="Speed change factor (0.5-3.0)"),
    caption_font_size: int = typer.Option(48, "--font-size", help="Caption font size"),
    trending_effects: bool = typer.Option(True, "--effects/--no-effects", help="Add trending visual effects"),
    auto_highlight: bool = typer.Option(True, "--auto-highlight/--no-auto-highlight", help="Auto-detect highlights for effects")
):
    """
    Transform a video into TikTok/YouTube Shorts style with mobile optimization.
    
    Examples:
        python main.py outputs/video.mp4 
        python main.py outputs/video.mp4 --output shorts/viral_video.mp4 --duration 30
        python main.py outputs/video.mp4 --aspect-ratio 1:1 --no-captions --zoom-intensity 1.5
    """
    
    # Validate input file
    input_path = Path(path)
    if not input_path.exists():
        console.print(f"[red]Error: Input file '{path}' not found[/red]")
        raise typer.Exit(1)
    
    # Set output path
    if output is None:
        output_dir = input_path.parent / "shorts"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_shorts{input_path.suffix}"
    else:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create editor configuration
    config = ShortsConfig(
        aspect_ratio=aspect_ratio,
        target_duration=duration,
        add_captions=add_captions,
        add_zooms=add_zooms,
        speed_variations=speed_variations,
        add_music=add_music,
        text_style=text_style,
        zoom_intensity=zoom_intensity,
        speed_factor=speed_factor,
        caption_font_size=caption_font_size,
        trending_effects=trending_effects,
        auto_highlight=auto_highlight
    )
    
    console.print(f"[blue]🎬 Creating TikTok/YouTube Shorts from: {input_path}[/blue]")
    console.print(f"[cyan]📱 Target: {aspect_ratio} aspect ratio, {duration}s duration[/cyan]")
    
    # Initialize editor
    try:
        editor = ShortsVideoEditor(config)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Processing video...", total=None)
            
            # Process the video
            result = editor.create_shorts_video(input_path, output_path, progress_callback=lambda msg: progress.update(task, description=msg))
            
            if result:
                console.print(f"[green]✅ Successfully created shorts video: {output_path}[/green]")
                console.print(f"[yellow]📊 Stats:[/yellow]")
                console.print(f"  • Duration: {result.get('duration', 'N/A')}s")
                console.print(f"  • Resolution: {result.get('resolution', 'N/A')}")
                console.print(f"  • Captions: {result.get('caption_count', 0)} segments")
                console.print(f"  • Effects: {result.get('effects_applied', [])} ")
            else:
                console.print("[red]❌ Failed to create shorts video[/red]")
                raise typer.Exit(1)
                
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

def batch_process(
    input_dir: str = typer.Argument(..., help="Directory containing video files"),
    output_dir: str = typer.Argument(..., help="Output directory for shorts videos"),
    pattern: str = typer.Option("*.mp4", "--pattern", "-p", help="File pattern to match"),
    **kwargs
):
    """Batch process multiple videos into shorts format."""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        console.print(f"[red]Error: Input directory '{input_dir}' not found[/red]")
        raise typer.Exit(1)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find video files
    video_files = list(input_path.glob(pattern))
    
    if not video_files:
        console.print(f"[yellow]No video files found matching pattern '{pattern}'[/yellow]")
        return
    
    console.print(f"[blue]Found {len(video_files)} video files to process[/blue]")
    
    for video_file in video_files:
        output_file = output_path / f"{video_file.stem}_shorts{video_file.suffix}"
        console.print(f"\n[cyan]Processing: {video_file.name}[/cyan]")
        
        try:
            # Use the same logic as create_shorts but with different paths
            # Remove the path argument from kwargs since we're setting it
            filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'path'}
            create_shorts(str(video_file), str(output_file), **filtered_kwargs)
        except Exception as e:
            console.print(f"[red]Failed to process {video_file.name}: {e}[/red]")
            continue

def preview_effects(
    path: str = typer.Argument(..., help="Path to input video file"),
    effect_type: str = typer.Option("all", "--type", help="Effect type: captions, zoom, speed, all"),
    duration: float = typer.Option(10.0, "--duration", help="Preview duration in seconds")
):
    """Preview different effects on a video segment."""
    
    input_path = Path(path)
    if not input_path.exists():
        console.print(f"[red]Error: Input file '{path}' not found[/red]")
        raise typer.Exit(1)
    
    console.print(f"[blue]🎥 Generating {effect_type} effect preview for {duration}s...[/blue]")
    
    # This would generate a quick preview showing the effects
    # Implementation would be in the ShortsVideoEditor class
    try:
        config = ShortsConfig()
        editor = ShortsVideoEditor(config)
        preview_path = editor.generate_preview(input_path, effect_type, duration)
        console.print(f"[green]Preview generated: {preview_path}[/green]")
    except Exception as e:
        console.print(f"[red]Error generating preview: {e}[/red]")

# Create typer app and add commands
app = typer.Typer(help="TikTok/YouTube Shorts Style Video Editor")
app.command(name="create")(create_shorts)
app.command(name="batch")(batch_process)
app.command(name="preview")(preview_effects)

# Also make create_shorts the default command
app.command()(create_shorts)

if __name__ == "__main__":
    app()