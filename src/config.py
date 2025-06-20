"""Configuration settings for viral clip generator."""

import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import List, Optional

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
    MIN_CLIP_DURATION: int = 15  # seconds
    MAX_CLIP_DURATION: int = 60  # seconds
    TARGET_CLIP_DURATION: int = 30  # seconds
    
    # Model configurations
    WHISPER_MODEL: str = "large-v3"
    WHISPER_DEVICE: str = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
    
    # YOLO model
    YOLO_MODEL: str = "yolov8n.pt"
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

# Global config instance
config = Config()

# Create directories if they don't exist
for dir_path in [config.DATA_DIR, config.MODELS_DIR, config.OUTPUTS_DIR, config.TEMP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)