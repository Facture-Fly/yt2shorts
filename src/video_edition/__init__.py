"""
TikTok/YouTube Shorts Video Editor Module
Creates mobile-optimized viral videos with trending effects.
"""

from .config import ShortsConfig, TRENDING_EFFECTS, TRENDING_AUDIO

# Try to import video editors in order of preference
BACKEND_USED = None

# First try MoviePy (best features)
try:
    from .moviepy_editor import MoviePyShortsEditor as ShortsVideoEditor
    BACKEND_USED = "moviepy"
except ImportError:
    # Then try FFmpeg-python
    try:
        from .shorts_editor import ShortsVideoEditor
        BACKEND_USED = "ffmpeg"
    except ImportError:
        # Finally fallback to simple editor
        try:
            from .simple_editor import SimpleShortsEditor as ShortsVideoEditor
            BACKEND_USED = "simple"
        except ImportError:
            raise ImportError("No video processing backend available!")

# Legacy compatibility
USING_MOVIEPY = (BACKEND_USED == "moviepy")

__all__ = [
    'ShortsVideoEditor',
    'ShortsConfig', 
    'TRENDING_EFFECTS',
    'TRENDING_AUDIO',
    'USING_MOVIEPY'
]