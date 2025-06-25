#!/usr/bin/env python3
"""
Example usage of the TikTok/YouTube Shorts Video Editor
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from video_edition.config import ShortsConfig
from video_edition.shorts_editor import ShortsVideoEditor

def create_gaming_short(input_video: str, output_video: str):
    """Create a gaming-style short with neon effects."""
    config = ShortsConfig(
        aspect_ratio="9:16",
        target_duration=30,
        text_style="neon",
        zoom_intensity=1.4,
        speed_factor=2.0,
        trending_effects=True,
        caption_font_size=56
    )
    
    editor = ShortsVideoEditor(config)
    return editor.create_shorts_video(Path(input_video), Path(output_video))

def create_educational_short(input_video: str, output_video: str):
    """Create an educational-style short with clear captions."""
    config = ShortsConfig(
        aspect_ratio="9:16",
        target_duration=60,
        text_style="minimal",
        zoom_intensity=1.1,
        speed_variations=False,
        trending_effects=False,
        caption_font_size=52
    )
    
    editor = ShortsVideoEditor(config)
    return editor.create_shorts_video(Path(input_video), Path(output_video))

def create_lifestyle_short(input_video: str, output_video: str):
    """Create a lifestyle/travel-style short with warm colors."""
    config = ShortsConfig(
        aspect_ratio="4:5",  # Instagram portrait
        target_duration=45,
        text_style="sunset",
        zoom_intensity=1.2,
        add_music=True,
        trending_effects=True,
        caption_font_size=48
    )
    
    editor = ShortsVideoEditor(config)
    return editor.create_shorts_video(Path(input_video), Path(output_video))

def create_comedy_short(input_video: str, output_video: str):
    """Create a comedy-style short with colorful effects."""
    config = ShortsConfig(
        aspect_ratio="9:16",
        target_duration=30,
        text_style="colorful", 
        zoom_intensity=1.3,
        speed_factor=1.8,
        trending_effects=True,
        caption_font_size=54
    )
    
    editor = ShortsVideoEditor(config)
    return editor.create_shorts_video(Path(input_video), Path(output_video))

def batch_create_shorts(input_dir: str, output_dir: str):
    """Create shorts from all videos in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Input directory {input_dir} does not exist")
        return
    
    output_path.mkdir(exist_ok=True)
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(input_path.glob(f'*{ext}'))
    
    if not video_files:
        print("No video files found")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Default config for batch processing
    config = ShortsConfig(
        aspect_ratio="9:16",
        target_duration=45,
        text_style="bold",
        add_captions=False,  # Faster processing
        auto_highlight=False  # Faster processing
    )
    
    editor = ShortsVideoEditor(config)
    
    for i, video_file in enumerate(video_files, 1):
        output_file = output_path / f"{video_file.stem}_shorts{video_file.suffix}"
        print(f"\n[{i}/{len(video_files)}] Processing: {video_file.name}")
        
        try:
            result = editor.create_shorts_video(video_file, output_file)
            if result.get("success"):
                print(f"✅ Created: {output_file}")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"❌ Error: {e}")

def demo_all_styles(input_video: str):
    """Demonstrate all text styles with one video."""
    input_path = Path(input_video)
    
    if not input_path.exists():
        print(f"Input video {input_video} does not exist")
        return
    
    styles = ["bold", "minimal", "colorful", "neon", "sunset"]
    output_dir = input_path.parent / "style_demos"
    output_dir.mkdir(exist_ok=True)
    
    for style in styles:
        print(f"\n🎨 Creating {style} style demo...")
        
        config = ShortsConfig(
            aspect_ratio="9:16",
            target_duration=20,
            text_style=style,
            add_captions=False,  # Focus on visual style
            trending_effects=False
        )
        
        editor = ShortsVideoEditor(config)
        output_file = output_dir / f"{input_path.stem}_{style}_demo{input_path.suffix}"
        
        try:
            result = editor.create_shorts_video(input_path, output_file)
            if result.get("success"):
                print(f"✅ {style.title()} style: {output_file}")
            else:
                print(f"❌ Failed {style}: {result.get('error')}")
        except Exception as e:
            print(f"❌ Error with {style}: {e}")

if __name__ == "__main__":
    print("🎬 TikTok/YouTube Shorts Editor - Examples")
    print("=" * 50)
    
    # Check if we have test videos
    test_videos = list(Path("../../outputs").glob("*.mp4"))
    
    if test_videos:
        test_video = str(test_videos[0])
        print(f"Using test video: {test_video}")
        
        print("\n📱 Available functions:")
        print("1. create_gaming_short(input, output)")
        print("2. create_educational_short(input, output)")
        print("3. create_lifestyle_short(input, output)")
        print("4. create_comedy_short(input, output)")
        print("5. batch_create_shorts(input_dir, output_dir)")
        print("6. demo_all_styles(input_video)")
        
        print(f"\n🎯 Example usage:")
        print(f"python -c \"from examples import *; create_gaming_short('{test_video}', 'gaming_short.mp4')\"")
        print(f"python -c \"from examples import *; demo_all_styles('{test_video}')\"")
        
    else:
        print("No test videos found in ../../outputs/")
        print("Place some .mp4 files there to test the examples")