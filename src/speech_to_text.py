"""Speech-to-text processing using OpenAI Whisper with ROCm support."""

import os
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
        
        # Set ROCm environment variables
        if "cuda" in self.device.lower():
            os.environ["CUDA_VISIBLE_DEVICES"] = config.ROCM_VISIBLE_DEVICES or "0"
            os.environ["PYTORCH_HIP_ALLOC_CONF"] = config.PYTORCH_HIP_ALLOC_CONF
        
        self._load_model()
    
    def _get_best_device(self) -> str:
        """Determine the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        else:
            console.print("[yellow]CUDA/ROCm not available, using CPU[/yellow]")
            return "cpu"
    
    def _load_model(self):
        """Load the Whisper model."""
        try:
            console.print(f"[blue]Loading Whisper model: {self.model_name} on {self.device}[/blue]")
            self.model = whisper.load_model(self.model_name, device=self.device)
            console.print("[green]Whisper model loaded successfully[/green]")
        except Exception as e:
            console.print(f"[red]Error loading Whisper model: {e}[/red]")
            console.print("[yellow]Falling back to CPU[/yellow]")
            self.device = "cpu"
            self.model = whisper.load_model(self.model_name, device=self.device)
    
    def transcribe_audio(self, audio_path: Path, language: str = None) -> List[TranscriptSegment]:
        """Transcribe audio file and return segments with timestamps."""
        if not self.model:
            console.print("[red]Model not loaded[/red]")
            return []
        
        try:
            console.print(f"[blue]Transcribing audio: {audio_path}[/blue]")
            
            # Load and preprocess audio
            audio = whisper.load_audio(str(audio_path))
            audio = whisper.pad_or_trim(audio)
            
            # Make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            
            # Detect language if not specified
            if not language:
                _, probs = self.model.detect_language(mel)
                language = max(probs, key=probs.get)
                console.print(f"[dim]Detected language: {language}[/dim]")
            
            # Decode the audio with timestamps
            options = whisper.DecodingOptions(
                language=language,
                task="transcribe",
                temperature=0.0,
                best_of=1,
                beam_size=1,
                patience=None,
                length_penalty=None,
                suppress_tokens="-1",
                initial_prompt=None,
                condition_on_previous_text=True,
                fp16=torch.cuda.is_available(),
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
            )
            
            # Transcribe with word-level timestamps
            result = self.model.transcribe(
                str(audio_path),
                language=language,
                task="transcribe",
                verbose=False,
                word_timestamps=True,
                temperature=0.0
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
            return segments
            
        except Exception as e:
            console.print(f"[red]Transcription error: {e}[/red]")
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
                temperature=0.0
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
    
    def detect_speaker_changes(self, segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """Detect potential speaker changes based on silence gaps and text patterns."""
        if len(segments) < 2:
            return segments
        
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
                if (gap > 2.0 or 
                    (prev_segment.text.strip().endswith('?') and not segment.text.strip().endswith('?')) or
                    (not prev_segment.text.strip().endswith('?') and segment.text.strip().endswith('?'))):
                    
                    speaker_count += 1
                    current_speaker = f"Speaker_{speaker_count}"
            
            segment.speaker = current_speaker
            enhanced_segments.append(segment)
        
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