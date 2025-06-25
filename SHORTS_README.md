# 🎬 TikTok/YouTube Shorts Video Editor

Transform regular videos into viral TikTok/YouTube Shorts with mobile optimization, dynamic effects, and trending elements.

## 🚀 Quick Start

### **Option 1: Automatic Environment (Recommended)**
```bash
# Automatically activates conda environment and runs with MoviePy
python make_shorts.py outputs/video.mp4

# Custom settings
python make_shorts.py outputs/video.mp4 --output shorts/my_viral.mp4 --duration 30 --text-style neon
```

### **Option 2: Manual Environment Activation**
```bash
# Activate the viral conda environment first
conda activate viral

# Then run the editor (gets full MoviePy features)
python shorts_maker.py outputs/video.mp4
```

### **Option 3: Simple Mode (No Conda)**
```bash
# Runs with basic features (no conda environment needed)
python shorts_maker.py outputs/video.mp4
```

## 🎯 Features by Backend

| Backend | Features | Quality | Speed |
|---------|----------|---------|-------|
| **MoviePy** (viral env) | ✅ Full effects, zooms, speed changes, mobile optimization | 🔥 High | ⚡ Fast |
| **FFmpeg** | ✅ Professional processing, good effects | 🔥 High | ⚡ Fast |
| **Simple** | ✅ Basic conversion, metadata optimization | ⭐ Good | 🚀 Very Fast |

## 📱 Mobile Optimization Features

### **✅ Aspect Ratio Conversion**
- **9:16** - TikTok, Instagram Reels, YouTube Shorts (default)
- **1:1** - Instagram Square posts
- **4:5** - Instagram Portrait posts
- **16:9** - YouTube Shorts landscape

### **✅ Dynamic Visual Effects**
- **Smart Zoom**: Auto-zoom on exciting moments
- **Speed Variations**: Speed ramps and slow-motion
- **Flash Transitions**: Trending white flash effects
- **Content-Aware Processing**: Different effects based on video type

### **✅ Text Styling**
- **bold** - High contrast white/black (default)
- **minimal** - Clean, professional look
- **colorful** - Fun, engaging colors
- **neon** - Gaming/tech style
- **sunset** - Warm lifestyle colors

## 🎨 Usage Examples

### **Gaming Content**
```bash
python make_shorts.py gameplay.mp4 --text-style neon --zoom-intensity 1.4 --duration 30
```

### **Educational/Tutorial**
```bash
python make_shorts.py tutorial.mp4 --text-style minimal --no-speed --font-size 52 --duration 60
```

### **Entertainment/Comedy**
```bash
python make_shorts.py funny.mp4 --text-style colorful --speed-factor 1.8 --effects --duration 30
```

### **Lifestyle/Travel**
```bash
python make_shorts.py travel.mp4 --text-style sunset --aspect-ratio 4:5 --duration 45
```

### **Instagram Square**
```bash
python make_shorts.py video.mp4 --aspect-ratio 1:1 --duration 30 --text-style minimal
```

## ⚙️ All Options

```bash
python make_shorts.py VIDEO_FILE [OPTIONS]

Options:
  -o, --output PATH              Output file path
  -d, --duration SECONDS         Target duration (default: 60)
  --aspect-ratio RATIO           9:16, 16:9, 1:1, 4:5 (default: 9:16)
  --text-style STYLE             bold, minimal, colorful, neon, sunset
  --font-size SIZE               Caption font size (default: 48)
  --zoom-intensity FACTOR        Zoom effect intensity 1.0-2.0 (default: 1.2)
  --speed-factor FACTOR          Speed change factor 0.5-3.0 (default: 1.5)
  --no-captions                  Disable automatic captions
  --no-zooms                     Disable zoom effects
  --no-speed                     Disable speed variations
  --no-effects                   Disable trending effects
  --music                        Add background music
  --help                         Show help message
```

## 🔧 Setup & Dependencies

### **For Full Features (MoviePy)**
```bash
# Activate the viral conda environment
conda activate viral

# MoviePy and other dependencies are already installed in this environment
```

### **For Basic Features**
```bash
# No special setup needed - works with basic Python
```

### **For Advanced Features**
```bash
# Optional: Install additional dependencies
pip install moviepy ffmpeg-python opencv-python
```

## 📊 Output Quality

### **Resolution & Format**
- **TikTok/Reels**: 1080x1920 (9:16) @ 30fps
- **Instagram Square**: 1080x1080 (1:1) @ 30fps
- **High Quality**: H.264 codec, optimized bitrate
- **Mobile Optimized**: Perfect for phone viewing

### **File Sizes** (approximate)
- **15-second clip**: ~5-8 MB
- **30-second clip**: ~10-15 MB
- **60-second clip**: ~20-30 MB

## 🎯 Platform-Specific Tips

### **TikTok**
```bash
python make_shorts.py video.mp4 --duration 15 --text-style bold --effects
```

### **Instagram Reels**
```bash
python make_shorts.py video.mp4 --duration 30 --text-style colorful --aspect-ratio 9:16
```

### **YouTube Shorts**
```bash
python make_shorts.py video.mp4 --duration 60 --text-style minimal --font-size 52
```

### **Instagram Posts**
```bash
python make_shorts.py video.mp4 --duration 30 --aspect-ratio 1:1 --text-style sunset
```

## 🚨 Troubleshooting

### **"MoviePy not available" Error**
```bash
# Solution: Use the automatic environment wrapper
python make_shorts.py your_video.mp4

# Or manually activate environment
conda activate viral
python shorts_maker.py your_video.mp4
```

### **"No such command" Error**
```bash
# Make sure you're in the right directory
cd ~/Documents/viral/yt2shorts

# Use the correct script name
python make_shorts.py outputs/video.mp4  # ✅ Correct
python main.py outputs/video.mp4         # ❌ Incorrect
```

### **Poor Quality Output**
```bash
# Use higher resolution input videos
# Increase font size for better text readability
python make_shorts.py video.mp4 --font-size 56 --zoom-intensity 1.1
```

## 🎉 Success!

Your TikTok/YouTube Shorts editor is ready! The system automatically:

✅ **Converts to mobile format** (9:16 aspect ratio)  
✅ **Optimizes for viral engagement** (dynamic effects)  
✅ **Adds professional styling** (multiple text themes)  
✅ **Handles all technical details** (encoding, compression)

**Start creating viral content:**
```bash
python make_shorts.py outputs/your_video.mp4
```

Your optimized shorts will be saved in the `shorts/` directory! 🚀