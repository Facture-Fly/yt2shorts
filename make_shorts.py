#!/usr/bin/env python3
"""
TikTok/YouTube Shorts Maker - Conda Environment Wrapper
Automatically activates the viral conda environment and runs the video editor.
"""

import sys
import subprocess
import os
from pathlib import Path

def main():
    """Run the shorts maker in the viral conda environment."""
    
    # Path to the actual editor
    editor_path = Path(__file__).parent / "src" / "video_edition" / "run_editor.py"
    
    if not editor_path.exists():
        print(f"❌ Error: Editor not found at {editor_path}")
        return 1
    
    # Check if we're already in the viral environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    
    if conda_env == 'viral':
        # Already in viral environment, run directly
        cmd = [sys.executable, str(editor_path)] + sys.argv[1:]
        return subprocess.call(cmd)
    else:
        # Need to activate viral environment
        print("🔄 Activating viral conda environment...")
        
        # Build command to activate conda env and run the editor
        conda_cmd = [
            "bash", "-c", 
            f"source ~/miniconda/etc/profile.d/conda.sh && "
            f"conda activate viral && "
            f"python {editor_path} " + " ".join(f'"{arg}"' for arg in sys.argv[1:])
        ]
        
        try:
            return subprocess.call(conda_cmd)
        except KeyboardInterrupt:
            print("\n⏹️  Interrupted by user")
            return 1
        except Exception as e:
            print(f"❌ Error activating environment: {e}")
            print("\n💡 Manual activation:")
            print("   conda activate viral")
            print(f"   python {editor_path} {' '.join(sys.argv[1:])}")
            return 1

if __name__ == "__main__":
    sys.exit(main())