# 🎬 Viral YouTube Clip Generator

A complete AI-powered pipeline for creating viral clips from YouTube videos using state-of-the-art models optimized for AMD ROCm.

## ✨ Features

- **🎥 Video Download**: Download from YouTube and other platforms using `yt-dlp`
- **🎙️ Speech-to-Text**: High-accuracy transcription with OpenAI Whisper (large-v3)
- **👁️ Visual Analysis**: Object detection (YOLO), emotion recognition, and action detection
- **🔥 Virality Scoring**: AI-powered analysis to identify the most engaging moments
- **🎬 Automated Clip Assembly**: Professional clip creation with effects, subtitles, and optimization
- **⚡ AMD ROCm Support**: Optimized for AMD GPUs with fallback to CPU

## 🏗️ Architecture

```
YouTube URL → Video Downloader → Video Processor
                                       ↓
Audio ← Speech-to-Text ← Audio Extraction
  ↓                           ↓
Transcript Analysis    Visual Analysis
  ↓                           ↓
        Virality Scorer ← Frames/Objects/Emotions
              ↓
        Clip Assembly → Post-Processing → Viral Clips
```

## 🚀 Installation

### Prerequisites

- **Python 3.10+**
- **Conda** (Miniconda or Anaconda)
- **FFmpeg** (for video processing)
- **AMD ROCm** (for GPU acceleration) or **CUDA** (optional)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd viral
```

### Step 2: Create Conda Environment

```bash
# Create environment from provided configuration
conda env create -f environment.yml

# Activate environment
conda activate viral-clips
```

### Step 3: Install Additional Dependencies

```bash
# Install remaining dependencies
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
# Check if everything is working
python main.py config-info
```

## 💻 Usage

### Basic Usage

Generate viral clips from a YouTube video:

```bash
python main.py generate "https://youtube.com/watch?v=VIDEO_ID"
```

### Advanced Usage

```bash
# Custom output directory and clip count
python main.py generate "VIDEO_URL" --output ./my_clips --max-clips 5

# Specific resolution and style
python main.py generate "VIDEO_URL" --resolution 720p --style energetic

# Disable effects and subtitles
python main.py generate "VIDEO_URL" --no-effects --no-subtitles

# Process local video file
python main.py generate "/path/to/video.mp4"
```

### Batch Processing

Create a file with URLs/paths (one per line):

```bash
# urls.txt
https://youtube.com/watch?v=VIDEO1
https://youtube.com/watch?v=VIDEO2
/path/to/local/video.mp4
```

Process all videos:

```bash
python main.py batch urls.txt --output ./batch_clips
```

### Get Video Information

```bash
python main.py info "https://youtube.com/watch?v=VIDEO_ID"
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# ROCm Configuration
ROCM_VISIBLE_DEVICES=0
PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128

# Model Configuration
WHISPER_MODEL=large-v3
YOLO_MODEL=yolov8n.pt

# Processing Settings
MAX_VIDEO_DURATION=3600
TARGET_CLIP_DURATION=30
MIN_CLIP_DURATION=15
MAX_CLIP_DURATION=60
```

### Clip Styles

Available styles:
- **default**: Balanced settings with moderate effects
- **energetic**: High-energy with bold subtitles and strong effects
- **minimal**: Clean and simple with minimal effects

### Resolution Options

- **720p**: 1280x720 (faster processing)
- **1080p**: 1920x1080 (recommended)
- **4k**: 3840x2160 (highest quality, slower)

## 🎯 How It Works

### 1. Video Analysis
- Downloads video from YouTube or loads local file
- Extracts audio for speech-to-text processing
- Extracts frames for visual analysis

### 2. Content Understanding
- **Speech Analysis**: Transcribes audio with timestamp precision
- **Visual Analysis**: Detects objects, emotions, and actions
- **Speaker Detection**: Identifies speaker changes and dialogue

### 3. Virality Scoring
- Analyzes emotional peaks and engagement factors
- Identifies surprising moments and reactions
- Scores based on visual appeal and audio energy
- Combines multiple signals for optimal moment detection

### 4. Clip Generation
- Creates professional clips with effects and transitions
- Adds animated subtitles with custom styling
- Optimizes audio quality and visual appeal
- Generates eye-catching thumbnails

## 🛠️ Technical Details

### Models Used

- **Whisper large-v3**: Speech-to-text transcription
- **YOLOv8**: Object detection
- **CLIP**: Visual understanding
- **RoBERTa**: Sentiment analysis
- **Face Recognition**: Emotion detection

### Optimization Features

- **4-bit Quantization**: Reduces memory usage
- **Dynamic Memory Management**: Efficient GPU utilization
- **Batch Processing**: Process multiple videos efficiently
- **Streaming Optimization**: Fast-start video encoding

### Output Formats

- **Video**: MP4 (H.264/AAC, optimized for social media)
- **Subtitles**: SRT, VTT, and embedded
- **Thumbnails**: High-quality JPEG with overlays

## 📁 Project Structure

```
viral/
├── src/
│   ├── config.py           # Configuration management
│   ├── downloader.py       # Video downloading
│   ├── video_processor.py  # Video/audio processing
│   ├── speech_to_text.py   # Whisper transcription
│   ├── visual_analysis.py  # Visual AI analysis
│   ├── virality_scorer.py  # Virality detection
│   ├── clip_assembler.py   # Clip creation
│   └── pipeline.py         # Main pipeline
├── main.py                 # CLI interface
├── environment.yml         # Conda environment
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔧 Troubleshooting

### Common Issues

**1. ROCm Not Detected**
```bash
# Check ROCm installation
rocm-smi
# If not working, install ROCm or use CPU mode
```

**2. CUDA Memory Errors**
```bash
# Reduce batch size in config.py
BATCH_SIZE = 4  # Default is 8
```

**3. FFmpeg Not Found**
```bash
# Install FFmpeg
sudo apt install ffmpeg  # Ubuntu/Debian
brew install ffmpeg      # macOS
```

**4. Model Download Issues**
```bash
# Models will auto-download on first use
# Ensure stable internet connection
# Check disk space (models require ~5GB)
```

### Performance Tips

1. **Use GPU acceleration** for faster processing
2. **Lower resolution** (720p) for speed over quality
3. **Reduce max_clips** for quicker results
4. **Use batch processing** for multiple videos

## 📊 Example Output

```
🎬 VIRAL CLIP GENERATOR 🎬
Processing: https://youtube.com/watch?v=example

📺 Title: Amazing Funny Video
⏱️  Duration: 10.5 minutes
🎵 Audio extracted: example_audio.mp3
📝 Transcribed 45 segments
👁️  Visual analysis: 123 objects, 67 emotions detected
🔥 Found 8 viral moments

🔥 Top Viral Moments
┏━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Rank ┃ Time       ┃ Duration ┃ Score  ┃ Preview                                          ┃
┡━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1    │ 125.3s     │ 28.4s    │ 0.89   │ Oh my god, you won't believe what just happen... │
│ 2    │ 67.1s      │ 22.1s    │ 0.82   │ Wait, wait, wait! This is absolutely incredi... │
│ 3    │ 201.5s     │ 31.2s    │ 0.76   │ No way! Are you serious right now? This is...   │
└──────┴────────────┴──────────┴────────┴──────────────────────────────────────────────────┘

🎬 Generated 3 viral clips

✅ Generated Clips:
  1. /path/to/outputs/viral_clip_125.3s_0.89.mp4
  2. /path/to/outputs/viral_clip_67.1s_0.82.mp4
  3. /path/to/outputs/viral_clip_201.5s_0.76.mp4

🖼️  Generated Thumbnails:
  1. /path/to/outputs/thumbnail_125.3s.jpg
  2. /path/to/outputs/thumbnail_67.1s.jpg
  3. /path/to/outputs/thumbnail_201.5s.jpg
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI** for Whisper speech-to-text
- **Ultralytics** for YOLO object detection
- **Hugging Face** for transformer models
- **FFmpeg** for video processing
- **yt-dlp** for video downloading

---

**⚡ Powered by AI • Optimized for AMD ROCm • Built for Viral Content Creation**