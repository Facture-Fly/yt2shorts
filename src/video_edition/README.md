# TikTok/YouTube Shorts Video Editor

Transform regular videos into engaging, mobile-optimized TikTok/YouTube Shorts style content with automatic captions, dynamic effects, and trending visual elements.

## 🎬 Features

### ✅ **Mobile Optimization**
- **9:16 Aspect Ratio**: Perfect for TikTok, Instagram Reels, YouTube Shorts
- **Alternative Ratios**: Support for 1:1 (Instagram), 16:9 (YouTube), 4:5 (Instagram Portrait)
- **High Quality Output**: 1080p resolution with optimized bitrates

### ✅ **Automatic Captions & Text Overlays**
- **Speech-to-Text**: Automatic transcription using Whisper
- **Styled Captions**: Bold, minimal, colorful, neon, and sunset themes
- **Smart Positioning**: Bottom, top, or center placement
- **Multi-line Support**: Automatic text wrapping for readability

### ✅ **Dynamic Visual Effects**
- **Smart Zooms**: Auto-zoom on exciting moments and beat drops
- **Speed Variations**: Speed ramps and slow-motion effects
- **Flash Transitions**: Trending white flash effects between scenes
- **Content-Aware Effects**: Different effects based on video content type

### ✅ **Trending Elements**
- **Music Integration**: Background music with volume balancing
- **Beat Sync Effects**: Effects synchronized to audio beats
- **Color Pop**: Selective color highlighting
- **Text Animations**: Slide-up and fade-in text reveals

## 🚀 Quick Start

### Basic Usage
```bash
# Convert a video to TikTok format with default settings
python main.py outputs/video.mp4

# Custom duration and aspect ratio
python main.py outputs/video.mp4 --duration 30 --aspect-ratio 1:1

# Disable captions, increase zoom intensity
python main.py outputs/video.mp4 --no-captions --zoom-intensity 1.5

# Minimal style with custom output location
python main.py outputs/video.mp4 --output shorts/my_viral_video.mp4 --text-style minimal
```

### Advanced Options
```bash
# Full customization
python main.py outputs/video.mp4 \
  --duration 45 \
  --aspect-ratio 9:16 \
  --text-style colorful \
  --font-size 56 \
  --zoom-intensity 1.3 \
  --speed-factor 1.8 \
  --music \
  --effects

# Batch processing
python main.py batch-process outputs/ shorts_output/ --pattern "*.mp4"

# Preview effects
python main.py preview-effects outputs/video.mp4 --type zoom --duration 10
```

## 📱 Output Specifications

### Aspect Ratios & Resolutions
| Platform | Ratio | Resolution | Best For |
|----------|-------|------------|----------|
| TikTok | 9:16 | 1080×1920 | Mobile-first content |
| Instagram Reels | 9:16 | 1080×1920 | Vertical stories |
| YouTube Shorts | 9:16 | 1080×1920 | Mobile YouTube |
| Instagram Square | 1:1 | 1080×1080 | Feed posts |
| Instagram Portrait | 4:5 | 1080×1350 | Feed optimization |

### Text Styles
| Style | Description | Best For |
|-------|-------------|----------|
| `bold` | White text, black background, red accents | High contrast, clear readability |
| `minimal` | Dark text, subtle shadows | Clean, professional look |
| `colorful` | White text, colorful backgrounds | Fun, engaging content |
| `neon` | Cyan text, magenta accents | Gaming, tech content |
| `sunset` | Warm orange/gold colors | Lifestyle, travel content |

## 🎯 Content-Aware Processing

The editor automatically adapts its processing based on content type:

### Educational Content
- **Enhanced Audio**: Clear speech emphasis (1.3x weight)
- **Visual Aids**: Better visual processing (1.1x weight)
- **Moderate Effects**: Reduced distraction from learning

### Entertainment Content  
- **Emotion Focus**: Strong emotion detection (1.4x weight)
- **Action Emphasis**: Dynamic action effects (1.3x weight)
- **High Energy**: Maximum visual impact

### Conversation/Interview
- **Speaker Dynamics**: Advanced multi-speaker processing (1.4x weight)
- **Emotion Tracking**: Enhanced emotional arc detection
- **Minimal Visual Distraction**: Focus on dialogue

### Sports/Action
- **Visual Priority**: Maximum visual effect processing (1.3x weight)
- **Fast Cuts**: High-energy editing with rapid transitions
- **Motion Emphasis**: Enhanced action detection

## 🛠️ Installation & Dependencies

### Required Dependencies
```bash
# Core video processing
pip install ffmpeg-python

# Audio/visual analysis (optional but recommended)
pip install opencv-python numpy
pip install whisper torch
pip install ultralytics transformers

# UI and utilities
pip install rich typer
```

### System Requirements
- **FFmpeg**: Must be installed on system for video processing
- **Python**: 3.8+ recommended
- **Memory**: 4GB+ RAM for video processing
- **Storage**: Temporary space for processing (2x video size)

## 📊 Performance Tips

### Optimize Processing Speed
- Use shorter durations for faster processing
- Disable unused features (captions, effects, music)
- Process in batches for multiple videos
- Use SSD storage for temporary files

### Quality vs Speed Trade-offs
```bash
# Fast processing (lower quality)
python main.py video.mp4 --no-captions --no-effects --duration 15

# High quality (slower processing)
python main.py video.mp4 --font-size 64 --zoom-intensity 1.5 --effects --music
```

## 🎨 Customization Examples

### Gaming Content
```bash
python main.py gameplay.mp4 \
  --text-style neon \
  --zoom-intensity 1.4 \
  --speed-factor 2.0 \
  --effects \
  --duration 30
```

### Educational Tutorial
```bash
python main.py tutorial.mp4 \
  --text-style minimal \
  --font-size 52 \
  --no-speed \
  --zoom-intensity 1.1 \
  --duration 60
```

### Lifestyle/Travel
```bash
python main.py travel.mp4 \
  --text-style sunset \
  --aspect-ratio 4:5 \
  --music \
  --zoom-intensity 1.2 \
  --duration 45
```

### Comedy/Entertainment
```bash
python main.py funny.mp4 \
  --text-style colorful \
  --speed-factor 1.8 \
  --effects \
  --zoom-intensity 1.3 \
  --duration 30
```

## 🔧 Troubleshooting

### Common Issues

**"ffmpeg not found"**
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

**"Module not found" errors**
```bash
# Install missing dependencies
pip install ffmpeg-python opencv-python numpy rich typer
```

**"Out of memory" errors**
- Reduce video duration
- Lower resolution input videos
- Close other applications
- Use `--no-captions --no-effects` for minimal processing

**Poor quality output**
- Increase `--font-size` for better text readability
- Use higher resolution input videos
- Adjust `--zoom-intensity` if zooms are too aggressive

### Performance Issues
- Use SSD storage for faster I/O
- Process shorter clips (30-60 seconds)
- Disable unused features
- Monitor system memory usage

## 📈 Output Stats

After processing, the editor provides detailed statistics:
- **Duration**: Final video length
- **Resolution**: Output dimensions  
- **Captions**: Number of subtitle segments
- **Effects**: Applied visual effects
- **Processing Time**: Total time taken
- **File Size**: Output file size

## 🎯 Best Practices

### For Maximum Engagement
1. **Keep it Short**: 15-30 seconds for highest retention
2. **Strong Opening**: First 3 seconds are crucial
3. **Clear Text**: Use high contrast colors and readable fonts
4. **Dynamic Editing**: Mix zoom, speed, and transition effects
5. **Audio Balance**: Keep original audio clear with subtle music

### Platform-Specific Tips

**TikTok**
- Use trending sounds and effects
- 15-30 second duration optimal
- Bold, attention-grabbing text
- Quick cuts and transitions

**Instagram Reels**
- High-quality visuals essential
- 15-60 second range
- Aesthetic color schemes
- Smooth transitions

**YouTube Shorts** 
- Up to 60 seconds allowed
- Clear, readable captions important
- Hook viewers in first 3 seconds
- Strong call-to-action at end

---

**🎬 Happy creating! Transform your videos into viral shorts with just one command.**