import streamlit as st
import os
from datetime import datetime
from beast_gen import (
    download_video, get_transcription, 
    find_highlight_segments, create_vid_gear_short
)

# Configuration
INPUT_VIDEO_PATH = "input_video.mp4"
OUTPUT_VIDEO_PATH = "beast_short.mp4"
TRANSCRIPTION_CACHE_DIR = ".transcription_cache"

def main():
    st.set_page_config(page_title="Mr. Beast Style Shorts Creator", layout="wide")
    
    st.title("🎬 Mr. Beast Style Shorts Creator")
    st.markdown("Transform long videos into engaging, fast-paced shorts using Mr. Beast-style editing!")

    # File upload/URL input section
    with st.container(border=True):
        input_tab, url_tab = st.tabs(["Upload Video", "YouTube URL"])
        
        with input_tab:
            uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
            
        with url_tab:
            youtube_url = st.text_input("Enter YouTube URL:")
            if youtube_url:
                with st.spinner("Downloading YouTube video..."):
                    try:
                        download_video(youtube_url)
                        st.success("Video downloaded successfully!")
                    except Exception as e:
                        st.error(f"Download failed: {str(e)}")

    # Processing controls sidebar
    with st.sidebar:
        st.header("🎛️ Processing Parameters")
        
        # Highlight Detection Parameters
        with st.expander("🎯 Highlight Detection", expanded=True):
            top_n = st.slider("Number of highlights to extract", 1, 10, 5,
                            help="How many highlight segments to include in the short")
            min_duration = st.slider("Minimum highlight duration (seconds)", 3, 15, 5,
                                   help="Shortest allowed duration for a highlight segment")
            max_duration = st.slider("Maximum highlight duration (seconds)", 10, 30, 15,
                                   help="Longest allowed duration for a highlight segment")
            min_segment_gap = st.slider("Minimum gap between segments (seconds)", 30, 120, 60,
                                      help="Ensures variety by spacing out selected highlights")
            positional_boost = st.checkbox("Boost opening/ending segments", value=True,
                                         help="Prioritize content from the beginning and end of video")
        
        # Video Effects Parameters
        with st.expander("🎨 Video Effects", expanded=True):
            st.subheader("Zoom Effects")
            zoom_factor = st.slider("Zoom intensity", 0.0, 0.2, 0.08, 0.01,
                                  help="How much to zoom in/out during transitions")
            zoom_duration = st.slider("Zoom duration (frames)", 5, 30, 15,
                                    help="How many frames the zoom effect lasts")
            
            st.subheader("Face Tracking")
            enable_face_tracking = st.checkbox("Enable dynamic face tracking", value=True,
                                             help="Automatically keeps faces centered in frame")
            max_tracking_speed = st.slider("Max tracking speed (pixels/sec)", 100, 800, 400,
                                         help="Maximum speed for camera movement when tracking faces")
            smoothing_frames = st.slider("Smoothing buffer", 3, 10, 5,
                                       help="Number of frames to average for smooth tracking")
        
        # Text Overlay Parameters
        with st.expander("📝 Text Overlay", expanded=True):
            text_lines = []
            num_lines = st.number_input("Number of text lines", 0, 5, 2)
            
            for i in range(int(num_lines)):
                st.markdown(f"**Line {i+1}**")
                col1, col2 = st.columns(2)
                with col1:
                    text = st.text_input(f"Text {i+1}", 
                                       value=f"EPIC MOMENT {i+1}" if i < 2 else "",
                                       key=f"text_{i}")
                with col2:
                    font_size = st.slider(f"Font size {i+1}", 0.5, 3.0, 1.5, 0.1,
                                        key=f"fontsize_{i}")
                text_lines.append({"text": text, "fontScale": font_size})
            
            text_position = st.selectbox("Text position", ["Top", "Center", "Bottom"], index=0)
            text_bg_opacity = st.slider("Background opacity", 0, 255, 200,
                                      help="Opacity of text background (0=transparent, 255=opaque)")
            stroke_thickness = st.slider("Text stroke thickness", 1, 5, 3,
                                       help="Thickness of text outline")
        
        # Video Output Parameters
        with st.expander("🎞️ Output Settings", expanded=False):
            output_width = st.number_input("Output width", 720, 1920, 1080, 
                                         help="Width of the output video")
            output_height = st.number_input("Output height", 1280, 2560, 1920,
                                          help="Height of the output video")
            video_preset = st.selectbox("Encoding preset", 
                                      ["ultrafast", "superfast", "veryfast", "faster", "fast", 
                                       "medium", "slow", "slower", "veryslow"],
                                      index=5,
                                      help="Quality vs speed tradeoff (slower = better quality)")
            crf_value = st.slider("Quality (CRF)", 18, 28, 23,
                                help="Lower = better quality, larger file size")
            
        # Transcription Parameters
        with st.expander("🎤 Transcription Settings", expanded=False):
            whisper_model = st.selectbox("Whisper model", 
                                       ["tiny", "base", "small", "medium", "large"],
                                       index=1,
                                       help="Larger models are more accurate but slower")
            force_refresh = st.checkbox("Force re-transcription", value=False,
                                      help="Ignore cached transcription and create new one")
        
        st.markdown("---")
        process_button = st.button("✨ Generate Short", type="primary", use_container_width=True)

    # Main processing flow
    if process_button and (uploaded_file or os.path.exists(INPUT_VIDEO_PATH)):
        if uploaded_file:
            # Save uploaded file
            with st.spinner("Uploading video..."):
                with open(INPUT_VIDEO_PATH, "wb") as f:
                    f.write(uploaded_file.getbuffer())
        
        try:
            # Create parameter dictionaries
            highlight_params = {
                "top_n": top_n,
                "min_duration": min_duration,
                "max_duration": max_duration,
                "min_segment_gap": min_segment_gap,
                "positional_boost": positional_boost
            }
            
            video_params = {
                "zoom_factor": zoom_factor,
                "zoom_duration": zoom_duration,
                "enable_face_tracking": enable_face_tracking,
                "max_tracking_speed": max_tracking_speed,
                "smoothing_frames": smoothing_frames,
                "text_lines": text_lines,
                "text_position": text_position,
                "text_bg_opacity": text_bg_opacity,
                "stroke_thickness": stroke_thickness,
                "output_width": int(output_width),
                "output_height": int(output_height),
                "video_preset": video_preset,
                "crf_value": crf_value
            }
            
            # Transcription
            with st.status("Analyzing video content...", expanded=True) as status:
                st.write("🔍 Transcribing audio...")
                # Note: You'll need to modify get_transcription to accept model parameter
                segments = get_transcription(force_refresh=force_refresh)
                
                st.write("🎯 Detecting highlights...")
                highlights = find_highlight_segments(segments, **highlight_params)
                
                # Show detected highlights
                st.subheader("Detected Highlights")
                highlight_data = []
                for idx, highlight in enumerate(highlights):
                    start = highlight["start"]
                    end = highlight["end"]
                    score = highlight["score"]
                    duration = end - start
                    highlight_data.append({
                        "Rank": idx + 1,
                        "Start Time": datetime.utcfromtimestamp(start).strftime('%M:%S'),
                        "Duration": f"{duration:.1f}s",
                        "Score": f"{score:.1f}",
                        "Preview": highlight["text"][:50] + "..." if len(highlight["text"]) > 50 else highlight["text"]
                    })
                
                st.dataframe(highlight_data, use_container_width=True)
                status.update(label="Analysis complete!", state="complete")

            # Video processing
            with st.status("Creating short...", expanded=True) as status:
                st.write("🎞️ Assembling clips...")
                st.write("🎨 Applying effects...")
                
                # Note: You'll need to modify create_vid_gear_short to accept these parameters
                # For now, showing what parameters would be passed
                st.info(f"Processing with parameters:\n"
                       f"- Zoom: {zoom_factor} intensity, {zoom_duration} frames\n"
                       f"- Face tracking: {'Enabled' if enable_face_tracking else 'Disabled'}\n"
                       f"- Output: {output_width}x{output_height} @ CRF {crf_value}")
                
                create_vid_gear_short(highlights)
                
                status.update(label="Render complete!", state="complete")

            # Display results
            st.success("✅ Short video created successfully!")
            st.subheader("Your Mr. Beast-Style Short")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                # Show video stats
                if os.path.exists(OUTPUT_VIDEO_PATH):
                    file_size = os.path.getsize(OUTPUT_VIDEO_PATH) / (1024 * 1024)  # MB
                    st.metric("File Size", f"{file_size:.1f} MB")
                    
                    with open(OUTPUT_VIDEO_PATH, "rb") as f:
                        st.video(f.read(), format="video/mp4")
                
                    # Download button
                    with open(OUTPUT_VIDEO_PATH, "rb") as f:
                        st.download_button(
                            label="⬇️ Download Short",
                            data=f,
                            file_name=f"beast_short_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )

        except Exception as e:
            st.error(f"Processing failed: {str(e)}")
            st.exception(e)
    
    # Footer
    with st.container():
        st.markdown("---")
        st.markdown("💡 **Tips for best results:**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("- Use videos with clear speech and exciting moments")
            st.markdown("- Adjust highlight detection based on your content type")
        with col2:
            st.markdown("- Enable face tracking for interview-style videos")
            st.markdown("- Use higher quality settings for final exports")

if __name__ == "__main__":
    main()