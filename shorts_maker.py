#!/usr/bin/env python3
"""
Convenient wrapper for the TikTok/YouTube Shorts Video Editor
"""

import sys
import subprocess
from pathlib import Path

# Path to the actual editor
EDITOR_PATH = Path(__file__).parent / "src" / "video_edition" / "run_editor.py"

def main():
    """Pass all arguments to the actual editor."""
    if not EDITOR_PATH.exists():
        print(f"Error: Editor not found at {EDITOR_PATH}")
        return 1
    
    # Pass all arguments to the real editor
    cmd = [sys.executable, str(EDITOR_PATH)] + sys.argv[1:]
    return subprocess.call(cmd)

if __name__ == "__main__":
    sys.exit(main())