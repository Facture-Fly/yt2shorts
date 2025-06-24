"""Configuration settings for viral clip generator."""

import os
from pathlib import Path
from typing import List, Optional

try:
    from pydantic import Field
    from pydantic_settings import BaseSettings
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseSettings = object
    Field = lambda default=None, **kwargs: default

if PYDANTIC_AVAILABLE:
    class Config(BaseSettings):
        # Paths
        BASE_DIR: Path = Path(__file__).parent.parent
        DATA_DIR: Path = BASE_DIR / "data"
        MODELS_DIR: Path = BASE_DIR / "models"
        OUTPUTS_DIR: Path = BASE_DIR / "outputs"
        TEMP_DIR: Path = BASE_DIR / "temp"
        
        # Video processing
        VIDEO_FORMAT: str = "mp4"
        AUDIO_FORMAT: str = "mp3"
        FRAME_RATE: int = 1  # frames per second for extraction
        MAX_VIDEO_DURATION: int = 3600  # 1 hour max
        
        # Clip generation
        MIN_CLIP_DURATION: int = 60  # seconds  
        MAX_CLIP_DURATION: int = 120  # seconds
        TARGET_CLIP_DURATION: int = 90  # seconds
        
        # TikTok-style viral editing
        VIRAL_SEGMENT_MIN_DURATION: float = 2.0  # seconds
        VIRAL_SEGMENT_MAX_DURATION: float = 8.0  # seconds
        VIRAL_SEGMENT_TARGET_DURATION: float = 4.0  # seconds
        MAX_VIRAL_SEGMENTS: int = 20  # Maximum segments per clip
        TRANSITION_DURATION: float = 0.3  # seconds
        
        # Model configurations
        WHISPER_MODEL: str = "base"
        WHISPER_DEVICE: str = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
        
        # YOLO model
        YOLO_MODEL: str = str(BASE_DIR / "models" / "yolov8n.pt")
        YOLO_CONFIDENCE: float = 0.5
        
        # LLM for virality scoring
        LLM_MODEL: str = "microsoft/DialoGPT-medium"
        LLM_MAX_LENGTH: int = 512
        LLM_TEMPERATURE: float = 0.7
        
        # ROCm settings
        ROCM_VISIBLE_DEVICES: Optional[str] = os.getenv("ROCM_VISIBLE_DEVICES", "0")
        PYTORCH_HIP_ALLOC_CONF: str = "max_split_size_mb:128"
        
        # Processing settings
        BATCH_SIZE: int = 8
        NUM_WORKERS: int = 4
        MAX_MEMORY_GB: int = 8
        
        # Virality criteria weights
        EMOTION_WEIGHT: float = 0.3
        ACTION_WEIGHT: float = 0.25
        AUDIO_WEIGHT: float = 0.2
        VISUAL_WEIGHT: float = 0.25
        
        # Video quality
        OUTPUT_RESOLUTION: str = "1080p"  # 720p, 1080p, 4k
        OUTPUT_BITRATE: str = "2M"
        OUTPUT_FPS: int = 30
        
        class Config:
            env_file = ".env"
            case_sensitive = True
else:
    class Config:
        def __init__(self):
            # Paths
            self.BASE_DIR = Path(__file__).parent.parent
            self.DATA_DIR = self.BASE_DIR / "data"
            self.MODELS_DIR = self.BASE_DIR / "models"
            self.OUTPUTS_DIR = self.BASE_DIR / "outputs"
            self.TEMP_DIR = self.BASE_DIR / "temp"
            
            # Video processing
            self.VIDEO_FORMAT = "mp4"
            self.AUDIO_FORMAT = "mp3"
            self.FRAME_RATE = 1
            self.MAX_VIDEO_DURATION = 3600
            
            # Clip generation
            self.MIN_CLIP_DURATION = 60
            self.MAX_CLIP_DURATION = 120
            self.TARGET_CLIP_DURATION = 90
            
            # TikTok-style viral editing
            self.VIRAL_SEGMENT_MIN_DURATION = 2.0
            self.VIRAL_SEGMENT_MAX_DURATION = 8.0
            self.VIRAL_SEGMENT_TARGET_DURATION = 4.0
            self.MAX_VIRAL_SEGMENTS = 20
            self.TRANSITION_DURATION = 0.3
            
            # Model configurations
            self.WHISPER_MODEL = "base"
            self.WHISPER_DEVICE = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
            
            # YOLO model
            self.YOLO_MODEL = str(self.BASE_DIR / "models" / "yolov8n.pt")
            self.YOLO_CONFIDENCE = 0.5
            
            # LLM for virality scoring
            self.LLM_MODEL = "microsoft/DialoGPT-medium"
            self.LLM_MAX_LENGTH = 512
            self.LLM_TEMPERATURE = 0.7
            
            # ROCm settings
            self.ROCM_VISIBLE_DEVICES = os.getenv("ROCM_VISIBLE_DEVICES", "0")
            self.PYTORCH_HIP_ALLOC_CONF = "max_split_size_mb:128"
            
            # Processing settings
            self.BATCH_SIZE = 8
            self.NUM_WORKERS = 4
            self.MAX_MEMORY_GB = 8
            
            # Virality criteria weights
            self.EMOTION_WEIGHT = 0.3
            self.ACTION_WEIGHT = 0.25
            self.AUDIO_WEIGHT = 0.2
            self.VISUAL_WEIGHT = 0.25
            
            # Video quality
            self.OUTPUT_RESOLUTION = "1080p"
            self.OUTPUT_BITRATE = "2M"
            self.OUTPUT_FPS = 30

# Global config instance
config = Config()

# Create directories if they don't exist
for dir_path in [config.DATA_DIR, config.MODELS_DIR, config.OUTPUTS_DIR, config.TEMP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)