#!/usr/bin/env python3
"""
Simple TikTok/YouTube Shorts Video Editor with Typer
Works like a normal CLI tool without subcommands.
"""

import sys
import typer
from pathlib import Path
from typing import Optional
from rich.console import Console

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from video_edition.config import ShortsConfig
from video_edition.shorts_editor import ShortsVideoEditor

console = Console()

def main(
    path: str = typer.Argument(..., help="Path to input video file"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    duration: float = typer.Option(60.0, "--duration", "-d", help="Target duration in seconds"),
    aspect_ratio: str = typer.Option("9:16", "--aspect-ratio", "-ar", help="Aspect ratio (9:16, 16:9, 1:1)"),
    no_captions: bool = typer.Option(False, "--no-captions", help="Disable automatic captions"),
    no_zooms: bool = typer.Option(False, "--no-zooms", help="Disable zoom effects"),
    no_speed: bool = typer.Option(False, "--no-speed", help="Disable speed variations"),
    music: bool = typer.Option(False, "--music", help="Add background music"),
    text_style: str = typer.Option("bold", "--text-style", help="Caption style: bold, minimal, colorful"),
    zoom_intensity: float = typer.Option(1.2, "--zoom-intensity", help="Zoom effect intensity (1.0-2.0)"),
    speed_factor: float = typer.Option(1.5, "--speed-factor", help="Speed change factor (0.5-3.0)"),
    font_size: int = typer.Option(48, "--font-size", help="Caption font size"),
    no_effects: bool = typer.Option(False, "--no-effects", help="Disable trending effects"),
    no_auto_highlight: bool = typer.Option(False, "--no-auto-highlight", help="Disable auto-highlight detection")
):
    """
    Transform a video into TikTok/YouTube Shorts style with mobile optimization.
    
    Examples:
        python main_simple.py outputs/video.mp4 
        python main_simple.py outputs/video.mp4 --output shorts/viral_video.mp4 --duration 30
        python main_simple.py outputs/video.mp4 --aspect-ratio 1:1 --no-captions --zoom-intensity 1.5
    """
    
    console.print("🎬 TikTok/YouTube Shorts Video Editor")
    console.print("=" * 50)
    
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
        add_captions=not no_captions,
        add_zooms=not no_zooms,
        speed_variations=not no_speed,
        add_music=music,
        text_style=text_style,
        zoom_intensity=zoom_intensity,
        speed_factor=speed_factor,
        caption_font_size=font_size,
        trending_effects=not no_effects,
        auto_highlight=not no_auto_highlight
    )
    
    console.print(f"📹 Input: {input_path}")
    console.print(f"📱 Output: {output_path}")  
    console.print(f"⚙️  Settings: {aspect_ratio}, {duration}s, {text_style} style")
    
    # Initialize editor
    try:
        editor = ShortsVideoEditor(config)
        
        console.print("\n🔄 Processing video...")
        
        # Process the video
        result = editor.create_shorts_video(
            input_path, 
            output_path, 
            progress_callback=lambda msg: console.print(f"   {msg}")
        )
        
        if result.get("success"):
            console.print(f"\n✅ Successfully created shorts video!")
            console.print(f"📊 Stats:")
            console.print(f"   • Output: {output_path}")
            console.print(f"   • Duration: {result.get('duration', 'N/A')}s")
            console.print(f"   • Resolution: {result.get('resolution', 'N/A')}")
            console.print(f"   • Captions: {result.get('caption_count', 0)} segments")
            
            effects = result.get('effects_applied', [])
            if any(effects):
                effect_list = [effect for sublist in effects for effect in sublist]
                unique_effects = list(set(effect_list))
                console.print(f"   • Effects: {', '.join(unique_effects) if unique_effects else 'None'}")
        else:
            console.print(f"\n❌ Failed to create shorts video: {result.get('error', 'Unknown error')}")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"\n❌ Error: {e}")
        raise typer.Exit(1)

if __name__ == "__main__":
    typer.run(main)