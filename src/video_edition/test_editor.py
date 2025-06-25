#!/usr/bin/env python3
"""
Test script for TikTok/YouTube Shorts Video Editor
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from video_edition.config import ShortsConfig
from video_edition.shorts_editor import ShortsVideoEditor

def test_config():
    """Test configuration system."""
    print("🧪 Testing configuration...")
    
    config = ShortsConfig()
    print(f"✅ Default aspect ratio: {config.aspect_ratio}")
    print(f"✅ Output resolution: {config.output_resolution}")
    print(f"✅ Color scheme: {config.get_current_color_scheme()}")
    
    # Test custom aspect ratio
    config_custom = ShortsConfig(aspect_ratio="1:1")
    print(f"✅ Custom 1:1 resolution: {config_custom.output_resolution}")

def test_editor_init():
    """Test editor initialization."""
    print("\n🧪 Testing editor initialization...")
    
    config = ShortsConfig(add_captions=False, auto_highlight=False)
    editor = ShortsVideoEditor(config)
    
    print(f"✅ Editor initialized with temp dir: {editor.temp_dir}")
    print(f"✅ Config loaded: {editor.config.aspect_ratio}")

def main():
    """Run all tests."""
    print("🎬 TikTok/YouTube Shorts Video Editor - Test Suite")
    print("=" * 60)
    
    try:
        test_config()
        test_editor_init()
        
        print("\n🎉 All tests passed! Editor is ready to use.")
        print("\nUsage examples:")
        print("python main.py outputs/video.mp4")
        print("python main.py outputs/video.mp4 --duration 30 --aspect-ratio 1:1")
        print("python main.py outputs/video.mp4 --no-captions --zoom-intensity 1.5")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())