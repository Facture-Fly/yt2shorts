"""
Simple caption system for MoviePy editor that works with minimal dependencies.
"""

import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from rich.console import Console

console = Console()

@dataclass
class SimpleTranscriptSegment:
    """Simple transcript segment."""
    start: float
    end: float
    text: str
    confidence: float = 0.8

class SimpleCaptionGenerator:
    """Simple caption generator using basic speech-to-text."""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="simple_captions_"))
        console.print("[cyan]Simple caption generator initialized[/cyan]")
    
    def extract_audio_and_transcribe(self, video_clip) -> List[SimpleTranscriptSegment]:
        """Extract audio from video and generate simple transcript."""
        try:
            # Try to use whisper if available
            try:
                import whisper
                return self._whisper_transcribe(video_clip)
            except ImportError:
                console.print("[yellow]Whisper not available, trying alternative methods[/yellow]")
                return self._fallback_transcribe(video_clip)
                
        except Exception as e:
            console.print(f"[yellow]Caption generation failed: {e}[/yellow]")
            return []
    
    def _whisper_transcribe(self, video_clip) -> List[SimpleTranscriptSegment]:
        """Use Whisper for transcription."""
        try:
            import whisper
            
            # Extract audio to temporary file
            audio_path = self.temp_dir / "audio.wav"
            video_clip.audio.write_audiofile(str(audio_path), logger=None, verbose=False)
            
            # Load whisper model (smallest one for speed)
            console.print("[blue]Loading Whisper model for captions...[/blue]")
            model = whisper.load_model("base")
            
            # Transcribe
            result = model.transcribe(str(audio_path), word_timestamps=True)
            
            segments = []
            for segment in result["segments"]:
                segments.append(SimpleTranscriptSegment(
                    start=segment["start"],
                    end=segment["end"],
                    text=segment["text"].strip(),
                    confidence=0.8
                ))
            
            console.print(f"[green]Whisper generated {len(segments)} caption segments[/green]")
            return segments
            
        except Exception as e:
            console.print(f"[yellow]Whisper transcription failed: {e}[/yellow]")
            return []
    
    def _fallback_transcribe(self, video_clip) -> List[SimpleTranscriptSegment]:
        """Fallback: create placeholder captions."""
        try:
            duration = video_clip.duration
            
            # Create simple placeholder captions
            segments = []
            segment_duration = 3.0  # 3-second segments
            
            for i in range(int(duration / segment_duration)):
                start_time = i * segment_duration
                end_time = min(start_time + segment_duration, duration)
                
                # Create placeholder text
                text = f"Caption {i+1}"
                
                segments.append(SimpleTranscriptSegment(
                    start=start_time,
                    end=end_time,
                    text=text,
                    confidence=0.5
                ))
            
            console.print(f"[yellow]Generated {len(segments)} placeholder captions[/yellow]")
            return segments
            
        except Exception as e:
            console.print(f"[red]Fallback transcription failed: {e}[/red]")
            return []
    
    def __del__(self):
        """Cleanup temporary files."""
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)