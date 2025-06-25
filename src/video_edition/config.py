"""Configuration for TikTok/YouTube Shorts video editor."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path

@dataclass
class ShortsConfig:
    """Configuration for shorts video creation."""
    
    # Video format settings
    aspect_ratio: str = "9:16"  # Mobile-first format
    target_duration: float = 60.0  # seconds
    output_resolution: Tuple[int, int] = (1080, 1920)  # 9:16 at 1080p
    fps: int = 30
    bitrate: str = "3M"
    
    # Text and captions
    add_captions: bool = True
    caption_font_size: int = 48
    caption_font: str = "Arial-Bold"
    text_style: str = "bold"  # bold, minimal, colorful
    caption_position: str = "bottom"  # top, center, bottom
    caption_margin: int = 100  # pixels from edge
    subtitle_duration_buffer: float = 0.1  # seconds
    
    # Visual effects
    add_zooms: bool = True
    zoom_intensity: float = 1.2  # 1.0 = no zoom, 2.0 = 2x zoom
    zoom_duration: float = 2.0  # seconds per zoom
    speed_variations: bool = True
    speed_factor: float = 1.5  # speed multiplier for dynamic segments
    trending_effects: bool = True
    auto_highlight: bool = True
    
    # Audio settings
    add_music: bool = False
    music_volume: float = 0.3  # 0.0 to 1.0
    original_audio_volume: float = 0.8
    
    # Color schemes for different styles
    color_schemes: Dict[str, Dict[str, str]] = None
    
    # Trending effects settings
    flash_effect_intensity: float = 0.3
    transition_duration: float = 0.3
    beat_sync: bool = True
    
    def __post_init__(self):
        """Initialize default color schemes and validate settings."""
        if self.color_schemes is None:
            self.color_schemes = {
                "bold": {
                    "text_color": "#FFFFFF",
                    "background_color": "#000000",
                    "accent_color": "#FF6B6B",
                    "shadow_color": "#333333"
                },
                "minimal": {
                    "text_color": "#2C3E50",
                    "background_color": "#FFFFFF",
                    "accent_color": "#3498DB",
                    "shadow_color": "#BDC3C7"
                },
                "colorful": {
                    "text_color": "#FFFFFF",
                    "background_color": "#FF6B6B",
                    "accent_color": "#4ECDC4",
                    "shadow_color": "#45B7B8"
                },
                "neon": {
                    "text_color": "#00FFFF",
                    "background_color": "#000000",
                    "accent_color": "#FF00FF",
                    "shadow_color": "#333333"
                },
                "sunset": {
                    "text_color": "#FFFFFF",
                    "background_color": "#FF7F50",
                    "accent_color": "#FFD700",
                    "shadow_color": "#FF6347"
                }
            }
        
        # Set resolution based on aspect ratio
        self._set_resolution_from_aspect_ratio()
        
        # Validate settings
        self._validate_settings()
    
    def _set_resolution_from_aspect_ratio(self):
        """Set output resolution based on aspect ratio."""
        aspect_ratios = {
            "9:16": (1080, 1920),  # TikTok/Instagram Reels
            "16:9": (1920, 1080),  # YouTube Shorts landscape
            "1:1": (1080, 1080),   # Instagram square
            "4:5": (1080, 1350),   # Instagram portrait
            "3:4": (1080, 1440)    # Pinterest/Snapchat
        }
        
        if self.aspect_ratio in aspect_ratios:
            self.output_resolution = aspect_ratios[self.aspect_ratio]
        else:
            # Try to parse custom aspect ratio like "9:16"
            try:
                w_ratio, h_ratio = map(int, self.aspect_ratio.split(':'))
                # Scale to 1080 width for quality
                width = 1080
                height = int(width * h_ratio / w_ratio)
                self.output_resolution = (width, height)
            except:
                # Fallback to 9:16
                self.output_resolution = (1080, 1920)
    
    def _validate_settings(self):
        """Validate configuration settings."""
        # Clamp values to reasonable ranges
        self.zoom_intensity = max(1.0, min(3.0, self.zoom_intensity))
        self.speed_factor = max(0.3, min(5.0, self.speed_factor))
        self.caption_font_size = max(24, min(120, self.caption_font_size))
        self.music_volume = max(0.0, min(1.0, self.music_volume))
        self.original_audio_volume = max(0.0, min(1.0, self.original_audio_volume))
        self.target_duration = max(5.0, min(300.0, self.target_duration))
    
    def get_current_color_scheme(self) -> Dict[str, str]:
        """Get the current color scheme based on text_style."""
        return self.color_schemes.get(self.text_style, self.color_schemes["bold"])
    
    def get_ffmpeg_video_filters(self) -> List[str]:
        """Generate FFmpeg video filter strings for the current config."""
        filters = []
        
        # Scale and crop to target aspect ratio
        width, height = self.output_resolution
        filters.append(f"scale={width}:{height}:force_original_aspect_ratio=increase")
        filters.append(f"crop={width}:{height}")
        
        return filters
    
    def get_caption_style_css(self) -> str:
        """Generate CSS-like styling for captions."""
        colors = self.get_current_color_scheme()
        
        styles = {
            "bold": f"""
                font_size={self.caption_font_size}
                font_color={colors['text_color']}
                font='{self.caption_font}'
                box=1
                boxcolor={colors['background_color']}@0.8
                boxborderw=5
            """,
            "minimal": f"""
                font_size={self.caption_font_size}
                font_color={colors['text_color']}
                font='{self.caption_font}'
                shadowcolor={colors['shadow_color']}
                shadowx=2
                shadowy=2
            """,
            "colorful": f"""
                font_size={self.caption_font_size}
                font_color={colors['text_color']}
                font='{self.caption_font}'
                box=1
                boxcolor={colors['background_color']}@0.9
                boxborderw=8
                bordercolor={colors['accent_color']}
            """
        }
        
        return styles.get(self.text_style, styles["bold"])

# Predefined trending effects templates
TRENDING_EFFECTS = {
    "zoom_punch": {
        "description": "Quick zoom-in on beat drops",
        "zoom_factor": 1.3,
        "duration": 0.5,
        "easing": "ease_out"
    },
    "speed_ramp": {
        "description": "Speed up to slow motion transition",
        "speed_sequence": [1.0, 2.0, 0.5, 1.0],
        "durations": [1.0, 0.5, 1.0, 1.0]
    },
    "flash_transition": {
        "description": "White flash between scenes",
        "flash_duration": 0.1,
        "flash_intensity": 0.8
    },
    "text_reveal": {
        "description": "Animated text appearance",
        "animation": "slide_up",
        "duration": 0.8
    },
    "color_pop": {
        "description": "Selective color highlighting",
        "target_colors": ["red", "blue", "yellow"],
        "intensity": 1.5
    }
}

# Audio library for trending sounds (placeholder - would need actual audio files)
TRENDING_AUDIO = {
    "upbeat_1": {
        "path": "audio/trending/upbeat_dance.mp3",
        "bpm": 128,
        "mood": "energetic",
        "duration": 30
    },
    "chill_1": {
        "path": "audio/trending/chill_vibe.mp3", 
        "bpm": 85,
        "mood": "relaxed",
        "duration": 45
    },
    "dramatic_1": {
        "path": "audio/trending/dramatic_build.mp3",
        "bpm": 140,
        "mood": "intense", 
        "duration": 60
    }
}

# Font recommendations for different styles
RECOMMENDED_FONTS = {
    "bold": ["Arial-Bold", "Helvetica-Bold", "Impact", "Bebas-Neue"],
    "minimal": ["Arial", "Helvetica", "Roboto", "Open-Sans"],
    "colorful": ["Comic-Sans-MS", "Marker-Felt", "Chalkduster", "Noteworthy"],
    "elegant": ["Times-New-Roman", "Georgia", "Playfair-Display", "Crimson-Text"],
    "modern": ["Montserrat", "Proxima-Nova", "Source-Sans-Pro", "Lato"]
}