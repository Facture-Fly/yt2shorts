import streamlit as st
import os
from datetime import datetime
from beast_gen import (
    download_video, get_transcription, 
    find_highlight_segments, create_vid_gear_short, create_short
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
        st.header("Processing Parameters")
        
        with st.expander("Highlight Detection", expanded=True):
            top_n = st.slider("Number of highlights", 3, 10, 5)
            min_duration = st.slider("Minimum highlight duration (s)", 3, 15, 5)
            max_duration = st.slider("Maximum highlight duration (s)", 10, 30, 15)
            sentiment_weight = st.slider("Sentiment importance", 0.0, 2.0, 1.5)
        
        with st.expander("Video Style"):
            zoom_speed = st.slider("Zoom intensity", 0.0, 0.2, 0.08, 0.01)
            transition_type = st.selectbox("Transition style", ["slide", "fade", "zoom"])
            text_overlay = st.text_input("Custom text overlay", "EPIC CHALLENGE")
            
        process_button = st.button("✨ Generate Short", type="primary")

    # Main processing flow
    if process_button and (uploaded_file or os.path.exists(INPUT_VIDEO_PATH)):
        if uploaded_file:
            # Save uploaded file
            with st.spinner("Uploading video..."):
                with open(INPUT_VIDEO_PATH, "wb") as f:
                    f.write(uploaded_file.getbuffer())
        
        try:
            # Transcription
            with st.status("Analyzing video content...", expanded=True) as status:
                st.write("🔍 Transcribing audio...")
                segments = get_transcription()
                
                st.write("🎯 Detecting highlights...")
                highlights = find_highlight_segments(
                    segments,
                    top_n=top_n,
                    min_duration=min_duration,
                    max_duration=max_duration
                )
                
                # Show detected highlights
                st.subheader("Detected Highlights")
                for idx, highlight in enumerate(highlights):
                    start = highlight["start"]
                    score = highlight["score"]
                    st.write(f"{idx+1}. At {datetime.utcfromtimestamp(start).strftime('%M:%S')} (score: {score:.1f})")
                
                status.update(label="Analysis complete!", state="complete")

            # Video processing
            with st.status("Creating short...", expanded=True) as status:
                st.write("🎞️ Assembling clips...")
                create_vid_gear_short(highlights)
                
                st.write("🎨 Applying final touches...")
                status.update(label="Render complete!", state="complete")

            # Display results
            st.subheader("Your Mr. Beast-Style Short")
            with st.container():
                # Create centered columns
                col1, col2, col3 = st.columns([1, 6, 1])
                with col2:
                    with open(OUTPUT_VIDEO_PATH, "rb") as f:
                        st.video(f.read(), format="video/mp4")
                
            # Download button
            with open(OUTPUT_VIDEO_PATH, "rb") as f:
                st.download_button(
                    label="Download Short",
                    data=f,
                    file_name="beast_short.mp4",
                    mime="video/mp4"
                )

        except Exception as e:
            st.error(f"Processing failed: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()