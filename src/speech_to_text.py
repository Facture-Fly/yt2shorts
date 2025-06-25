"""Speech-to-text processing using OpenAI Whisper with ROCm support."""

import os
import gc
import psutil
import torch
import whisper
import librosa
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from rich.console import Console
from rich.progress import Progress, track

from .config import config

# Try to import pyannote for advanced speaker diarization
try:
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

console = Console()

@dataclass
class TranscriptSegment:
    """Represents a segment of transcribed speech."""
    start: float
    end: float
    text: str
    confidence: float = 0.0
    speaker: Optional[str] = None

class SpeechToText:
    """Speech-to-text processor using Whisper with AMD ROCm support."""
    
    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or config.WHISPER_MODEL
        self.device = device or self._get_best_device()
        self.model = None
        self.diarization_pipeline = None
        
        # Set ROCm environment variables
        if "cuda" in self.device.lower():
            os.environ["CUDA_VISIBLE_DEVICES"] = config.ROCM_VISIBLE_DEVICES or "0"
            os.environ["PYTORCH_HIP_ALLOC_CONF"] = config.PYTORCH_HIP_ALLOC_CONF
        
        self._load_model()
        self._load_diarization_pipeline()
    
    def _get_best_device(self) -> str:
        """Determine the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        else:
            console.print("[yellow]CUDA/ROCm not available, using CPU[/yellow]")
            return "cpu"
    
    def _load_model(self):
        """Load the Whisper model from local directory if available."""
        try:
            console.print(f"[blue]Loading Whisper model: {self.model_name} on {self.device}[/blue]")
            
            # Try to load from local models directory first
            local_whisper_path = config.MODELS_DIR / "whisper" / f"{self.model_name}.pt"
            if local_whisper_path.exists():
                console.print(f"[cyan]Loading from local model: {local_whisper_path}[/cyan]")
                self.model = whisper.load_model(str(local_whisper_path), device=self.device)
            else:
                # Fallback to downloading model
                console.print("[yellow]Local model not found, downloading...[/yellow]")
                self.model = whisper.load_model(self.model_name, device=self.device)
            
            console.print("[green]Whisper model loaded successfully[/green]")
        except Exception as e:
            console.print(f"[red]Error loading Whisper model: {e}[/red]")
            console.print("[yellow]Falling back to CPU[/yellow]")
            self.device = "cpu"
            try:
                if local_whisper_path.exists():
                    self.model = whisper.load_model(str(local_whisper_path), device=self.device)
                else:
                    self.model = whisper.load_model(self.model_name, device=self.device)
            except Exception as e2:
                console.print(f"[red]Failed to load Whisper model on CPU: {e2}[/red]")
                raise
    
    def _load_diarization_pipeline(self):
        """Load speaker diarization pipeline if available."""
        if not PYANNOTE_AVAILABLE:
            console.print("[yellow]pyannote.audio not available, using basic speaker detection[/yellow]")
            return
        
        try:
            console.print("[blue]Loading speaker diarization pipeline[/blue]")
            # Load a pre-trained speaker diarization pipeline
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=False  # Set to True if you have HuggingFace auth
            )
            console.print("[green]Speaker diarization pipeline loaded[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not load diarization pipeline: {e}[/yellow]")
            console.print("[yellow]Falling back to basic speaker detection[/yellow]")
            self.diarization_pipeline = None
    
    def _check_memory_usage(self) -> bool:
        """Check if enough memory is available for transcription."""
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        # Whisper base model needs ~1GB, medium ~5GB, large ~10GB
        model_memory_req = {"tiny": 0.4, "base": 1.0, "small": 2.0, "medium": 5.0, "large": 10.0, "large-v2": 10.0, "large-v3": 10.0}
        required_gb = model_memory_req.get(self.model_name, 1.0)
        
        if available_gb < required_gb:
            console.print(f"[red]Insufficient memory: {available_gb:.1f}GB available, {required_gb}GB required[/red]")
            return False
        
        console.print(f"[green]Memory check passed: {available_gb:.1f}GB available[/green]")
        return True
    
    def transcribe_audio(self, audio_path: Path, language: str = None) -> List[TranscriptSegment]:
        """Transcribe audio file and return segments with timestamps."""
        if not self.model:
            console.print("[red]Model not loaded[/red]")
            return []
        
        # Check memory before processing
        if not self._check_memory_usage():
            console.print("[yellow]Forcing garbage collection and retrying...[/yellow]")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if not self._check_memory_usage():
                console.print("[red]Still insufficient memory after cleanup[/red]")
                return []
        
        try:
            console.print(f"[blue]Transcribing audio: {audio_path}[/blue]")
            
            # Transcribe with word-level timestamps and memory optimization
            result = self.model.transcribe(
                str(audio_path),
                language=language,
                task="transcribe",
                verbose=False,
                word_timestamps=True,
                temperature=0.0,
                fp16=False  # Disable FP16 to reduce memory usage
            )
            
            segments = []
            for segment in result["segments"]:
                segments.append(TranscriptSegment(
                    start=segment["start"],
                    end=segment["end"],
                    text=segment["text"].strip(),
                    confidence=getattr(segment, "avg_logprob", 0.0)
                ))
            
            console.print(f"[green]Transcription completed: {len(segments)} segments[/green]")
            
            # Clean up memory after transcription
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return segments
            
        except Exception as e:
            console.print(f"[red]Transcription error: {e}[/red]")
            # Clean up memory on error
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return []
    
    def transcribe_with_word_timestamps(self, audio_path: Path, language: str = None) -> Dict[str, Any]:
        """Transcribe with detailed word-level timestamps."""
        if not self.model:
            console.print("[red]Model not loaded[/red]")
            return {}
        
        try:
            result = self.model.transcribe(
                str(audio_path),
                language=language,
                task="transcribe",
                verbose=False,
                word_timestamps=True,
                temperature=0.0,
                fp16=False  # Disable FP16 to reduce memory usage
            )
            
            return {
                "text": result["text"],
                "language": result["language"],
                "segments": result["segments"],
                "words": self._extract_words_from_segments(result["segments"])
            }
            
        except Exception as e:
            console.print(f"[red]Word-level transcription error: {e}[/red]")
            return {}
    
    def _extract_words_from_segments(self, segments: List[Dict]) -> List[Dict]:
        """Extract individual words with timestamps from segments."""
        words = []
        for segment in segments:
            if "words" in segment:
                for word in segment["words"]:
                    words.append({
                        "word": word["word"].strip(),
                        "start": word["start"],
                        "end": word["end"],
                        "confidence": word.get("probability", 0.0)
                    })
        return words
    
    def detect_silence_segments(self, audio_path: Path, threshold: float = 0.01, 
                              min_duration: float = 0.5) -> List[Tuple[float, float]]:
        """Detect silence segments in audio."""
        try:
            y, sr = librosa.load(str(audio_path))
            
            # Calculate RMS energy
            frame_length = 2048
            hop_length = 512
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Convert frames to time
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
            
            # Find silence segments
            silence_mask = rms < threshold
            silence_segments = []
            
            start_silence = None
            for i, is_silent in enumerate(silence_mask):
                if is_silent and start_silence is None:
                    start_silence = times[i]
                elif not is_silent and start_silence is not None:
                    duration = times[i] - start_silence
                    if duration >= min_duration:
                        silence_segments.append((start_silence, times[i]))
                    start_silence = None
            
            # Handle case where audio ends with silence
            if start_silence is not None:
                duration = times[-1] - start_silence
                if duration >= min_duration:
                    silence_segments.append((start_silence, times[-1]))
            
            return silence_segments
            
        except Exception as e:
            console.print(f"[red]Silence detection error: {e}[/red]")
            return []
    
    def detect_speaker_changes(self, segments: List[TranscriptSegment], audio_path: Path = None) -> List[TranscriptSegment]:
        """Detect speaker changes using advanced diarization if available."""
        if len(segments) < 2:
            return segments
        
        # Try advanced diarization first
        if self.diarization_pipeline and audio_path and audio_path.exists():
            return self._advanced_speaker_diarization(segments, audio_path)
        
        # Fallback to basic speaker detection
        return self._basic_speaker_detection(segments)
    
    def _advanced_speaker_diarization(self, segments: List[TranscriptSegment], audio_path: Path) -> List[TranscriptSegment]:
        """Advanced speaker diarization using pyannote.audio."""
        try:
            console.print("[blue]Running advanced speaker diarization[/blue]")
            
            # Run diarization on the audio file
            diarization = self.diarization_pipeline(str(audio_path))
            
            # Create speaker timeline
            speaker_timeline = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_timeline.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker
                })
            
            # Map transcript segments to speakers
            enhanced_segments = []
            for segment in segments:
                segment_mid = (segment.start + segment.end) / 2
                
                # Find the speaker active at this segment's midpoint
                assigned_speaker = "Unknown"
                for speaker_turn in speaker_timeline:
                    if speaker_turn['start'] <= segment_mid <= speaker_turn['end']:
                        assigned_speaker = speaker_turn['speaker']
                        break
                
                # If no exact match, find the closest speaker
                if assigned_speaker == "Unknown":
                    min_distance = float('inf')
                    for speaker_turn in speaker_timeline:
                        distance = min(
                            abs(segment_mid - speaker_turn['start']),
                            abs(segment_mid - speaker_turn['end'])
                        )
                        if distance < min_distance:
                            min_distance = distance
                            assigned_speaker = speaker_turn['speaker']
                
                segment.speaker = assigned_speaker
                enhanced_segments.append(segment)
            
            # Count unique speakers
            unique_speakers = set(seg.speaker for seg in enhanced_segments)
            console.print(f"[green]Advanced diarization completed: {len(unique_speakers)} speakers detected[/green]")
            
            return enhanced_segments
            
        except Exception as e:
            console.print(f"[yellow]Advanced diarization failed: {e}[/yellow]")
            console.print("[yellow]Falling back to basic speaker detection[/yellow]")
            return self._basic_speaker_detection(segments)
    
    def _basic_speaker_detection(self, segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """Basic speaker detection based on silence gaps and text patterns."""
        enhanced_segments = []
        current_speaker = "Speaker_1"
        speaker_count = 1
        
        for i, segment in enumerate(segments):
            if i > 0:
                prev_segment = segments[i-1]
                gap = segment.start - prev_segment.end
                
                # Detect speaker change based on:
                # 1. Long silence gap (>2 seconds)
                # 2. Change in speaking style (questions vs statements)
                # 3. Significant change in speaking tempo
                prev_words = len(prev_segment.text.split())
                curr_words = len(segment.text.split())
                prev_wpm = prev_words / max((prev_segment.end - prev_segment.start) / 60, 0.01)
                curr_wpm = curr_words / max((segment.end - segment.start) / 60, 0.01)
                tempo_change = abs(curr_wpm - prev_wpm) / max(prev_wpm, 1)
                
                if (gap > 2.0 or 
                    (prev_segment.text.strip().endswith('?') and not segment.text.strip().endswith('?')) or
                    (not prev_segment.text.strip().endswith('?') and segment.text.strip().endswith('?')) or
                    tempo_change > 0.5):  # Significant tempo change
                    
                    speaker_count += 1
                    current_speaker = f"Speaker_{speaker_count}"
            
            segment.speaker = current_speaker
            enhanced_segments.append(segment)
        
        console.print(f"[green]Basic speaker detection completed: {speaker_count} speakers detected[/green]")
        return enhanced_segments
    
    def extract_key_phrases(self, segments: List[TranscriptSegment], min_word_count: int = 3) -> List[Dict[str, Any]]:
        """Extract key phrases and emotional moments from transcript."""
        key_phrases = []
        
        # Emotional indicators
        emotional_words = {
            'excitement': ['amazing', 'incredible', 'wow', 'unbelievable', 'fantastic', 'awesome'],
            'surprise': ['whoa', 'what', 'no way', 'seriously', 'really', 'oh my god'],
            'laughter': ['haha', 'lol', 'funny', 'hilarious', 'laugh'],
            'emphasis': ['definitely', 'absolutely', 'totally', 'completely', 'exactly']
        }
        
        for segment in segments:
            text_lower = segment.text.lower()
            words = text_lower.split()
            
            if len(words) < min_word_count:
                continue
            
            # Check for emotional content
            emotions = []
            for emotion, indicators in emotional_words.items():
                if any(indicator in text_lower for indicator in indicators):
                    emotions.append(emotion)
            
            # Check for questions (potential engagement)
            has_question = '?' in segment.text
            
            # Check for emphasis (caps, exclamation)
            has_emphasis = '!' in segment.text or any(word.isupper() for word in segment.text.split())
            
            if emotions or has_question or has_emphasis:
                key_phrases.append({
                    'text': segment.text,
                    'start': segment.start,
                    'end': segment.end,
                    'emotions': emotions,
                    'has_question': has_question,
                    'has_emphasis': has_emphasis,
                    'speaker': segment.speaker
                })
        
        return key_phrases
    
    def save_transcript(self, segments: List[TranscriptSegment], output_path: Path, 
                       format: str = "srt") -> bool:
        """Save transcript in various formats."""
        try:
            if format.lower() == "srt":
                self._save_srt(segments, output_path)
            elif format.lower() == "vtt":
                self._save_vtt(segments, output_path)
            elif format.lower() == "txt":
                self._save_txt(segments, output_path)
            else:
                console.print(f"[red]Unsupported format: {format}[/red]")
                return False
            
            console.print(f"[green]Transcript saved: {output_path}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Error saving transcript: {e}[/red]")
            return False
    
    def _save_srt(self, segments: List[TranscriptSegment], output_path: Path):
        """Save transcript in SRT format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                start_time = self._format_timestamp(segment.start)
                end_time = self._format_timestamp(segment.end)
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment.text}\n\n")
    
    def _save_vtt(self, segments: List[TranscriptSegment], output_path: Path):
        """Save transcript in WebVTT format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            for segment in segments:
                start_time = self._format_timestamp(segment.start, vtt=True)
                end_time = self._format_timestamp(segment.end, vtt=True)
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment.text}\n\n")
    
    def _save_txt(self, segments: List[TranscriptSegment], output_path: Path):
        """Save transcript as plain text."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for segment in segments:
                speaker_prefix = f"[{segment.speaker}] " if segment.speaker else ""
                f.write(f"{speaker_prefix}{segment.text}\n")
    
    def _format_timestamp(self, seconds: float, vtt: bool = False) -> str:
        """Format timestamp for subtitle files."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        if vtt:
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
        else:
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')

def transcribe_audio(audio_path: Path, model_name: str = None) -> List[TranscriptSegment]:
    """Convenience function to transcribe audio."""
    stt = SpeechToText(model_name)
    return stt.transcribe_audio(audio_path)