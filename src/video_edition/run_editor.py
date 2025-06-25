#!/usr/bin/env python3
"""
Simple TikTok/YouTube Shorts Video Editor Runner
Works without external dependencies for basic functionality.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from video_edition.config import ShortsConfig
    from video_edition import ShortsVideoEditor, BACKEND_USED
    
    backend_names = {
        "moviepy": "MoviePy (Full Features)",
        "ffmpeg": "FFmpeg-Python (Advanced)", 
        "simple": "Simple Editor (Basic)"
    }
    
    backend_name = backend_names.get(BACKEND_USED, BACKEND_USED)
    print(f"🎬 Using {backend_name} for video processing")
    
except ImportError as e:
    print(f"Error importing video editor: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TikTok/YouTube Shorts Style Video Editor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_editor.py outputs/video.mp4
  python run_editor.py outputs/video.mp4 --duration 30 --aspect-ratio 1:1
  python run_editor.py outputs/video.mp4 --output shorts/viral.mp4 --no-captions
        """
    )
    
    parser.add_argument("path", help="Path to input video file")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("-d", "--duration", type=float, default=60.0, 
                        help="Target duration in seconds (default: 60)")
    parser.add_argument("--aspect-ratio", default="9:16",
                        help="Aspect ratio: 9:16, 16:9, 1:1 (default: 9:16)")
    parser.add_argument("--no-captions", action="store_true",
                        help="Disable automatic captions")
    parser.add_argument("--no-zooms", action="store_true",
                        help="Disable zoom effects")
    parser.add_argument("--no-speed", action="store_true",
                        help="Disable speed variations")
    parser.add_argument("--music", action="store_true",
                        help="Add background music")
    parser.add_argument("--text-style", default="bold",
                        choices=["bold", "minimal", "colorful", "neon", "sunset"],
                        help="Caption style (default: bold)")
    parser.add_argument("--zoom-intensity", type=float, default=1.2,
                        help="Zoom effect intensity 1.0-2.0 (default: 1.2)")
    parser.add_argument("--speed-factor", type=float, default=1.5,
                        help="Speed change factor 0.5-3.0 (default: 1.5)")
    parser.add_argument("--font-size", type=int, default=48,
                        help="Caption font size (default: 48)")
    parser.add_argument("--no-effects", action="store_true",
                        help="Disable trending effects")
    parser.add_argument("--no-auto-highlight", action="store_true",
                        help="Disable auto-highlight detection")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    print("🎬 TikTok/YouTube Shorts Video Editor")
    print("=" * 50)
    
    args = parse_args()
    
    # Validate input file
    input_path = Path(args.path)
    if not input_path.exists():
        print(f"❌ Error: Input file '{args.path}' not found")
        return 1
    
    # Set output path
    if args.output is None:
        output_dir = input_path.parent / "shorts"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_shorts{input_path.suffix}"
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create configuration
    config = ShortsConfig(
        aspect_ratio=args.aspect_ratio,
        target_duration=args.duration,
        add_captions=not args.no_captions,
        add_zooms=not args.no_zooms,
        speed_variations=not args.no_speed,
        add_music=args.music,
        text_style=args.text_style,
        zoom_intensity=args.zoom_intensity,
        speed_factor=args.speed_factor,
        caption_font_size=args.font_size,
        trending_effects=not args.no_effects,
        auto_highlight=not args.no_auto_highlight
    )
    
    print(f"📹 Input: {input_path}")
    print(f"📱 Output: {output_path}")
    print(f"⚙️  Settings: {args.aspect_ratio}, {args.duration}s, {args.text_style} style")
    
    # Initialize editor
    try:
        editor = ShortsVideoEditor(config)
        
        # Process the video
        print("\n🔄 Processing video...")
        
        result = editor.create_shorts_video(
            input_path, 
            output_path, 
            progress_callback=lambda msg: print(f"   {msg}")
        )
        
        if result.get("success"):
            print(f"\n✅ Successfully created shorts video!")
            print(f"📊 Stats:")
            print(f"   • Output: {output_path}")
            print(f"   • Duration: {result.get('duration', 'N/A')}s")
            print(f"   • Resolution: {result.get('resolution', 'N/A')}")
            print(f"   • Captions: {result.get('caption_count', 0)} segments")
            
            effects = result.get('effects_applied', [])
            if any(effects):
                effect_list = [effect for sublist in effects for effect in sublist]
                unique_effects = list(set(effect_list))
                print(f"   • Effects: {', '.join(unique_effects) if unique_effects else 'None'}")
            
            return 0
        else:
            print(f"\n❌ Failed to create shorts video: {result.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())