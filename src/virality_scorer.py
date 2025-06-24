"""Virality scoring using local LLM with ROCm support."""

import os
import json
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
    pipeline, BitsAndBytesConfig
)
from sentence_transformers import SentenceTransformer
import numpy as np
from rich.console import Console
from rich.progress import Progress, track

from .config import config
from .speech_to_text import TranscriptSegment
from .visual_analysis import DetectedObject, EmotionDetection, ActionDetection

console = Console()

@dataclass
class ViralMoment:
    """Represents a potential viral moment in the video."""
    start_time: float
    end_time: float
    duration: float
    virality_score: float
    reasons: List[str]
    transcript_segments: List[TranscriptSegment]
    visual_features: Dict[str, Any]
    audio_features: Dict[str, Any]
    confidence: float

class ViralityScorer:
    """Score video segments for viral potential using local LLM."""
    
    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or "microsoft/DialoGPT-medium"
        self.device = device or self._get_best_device()
        self.tokenizer = None
        self.model = None
        self.sentiment_pipeline = None
        self.sentence_transformer = None
        
        # Set up ROCm environment
        if "cuda" in self.device.lower():
            os.environ["PYTORCH_HIP_ALLOC_CONF"] = config.PYTORCH_HIP_ALLOC_CONF
        
        self._load_models()
    
    def _get_best_device(self) -> str:
        """Determine the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        else:
            console.print("[yellow]CUDA/ROCm not available, using CPU[/yellow]")
            return "cpu"
    
    def _load_models(self):
        """Load language models for virality scoring."""
        try:
            console.print(f"[blue]Loading virality scoring models on {self.device}[/blue]")
            
            # Configure quantization for memory efficiency
            if "cuda" in self.device:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            else:
                quantization_config = None
            
            # Load sentiment analysis pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if "cuda" in self.device else -1
            )
            
            # Load sentence transformer for semantic similarity
            console.print("[blue]Loading sentence transformer[/blue]")
            self.sentence_transformer = SentenceTransformer(
                'all-MiniLM-L6-v2',
                device=self.device
            )
            
            # Load main LLM for text generation
            console.print(f"[blue]Loading main LLM: {self.model_name}[/blue]")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto" if "cuda" in self.device else None,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                low_cpu_mem_usage=True
            )
            
            console.print("[green]All virality models loaded successfully[/green]")
            
        except Exception as e:
            console.print(f"[red]Error loading virality models: {e}[/red]")
            console.print("[yellow]Falling back to simplified scoring[/yellow]")
    
    def score_transcript_segment(self, segment: TranscriptSegment) -> Dict[str, Any]:
        """Score a single transcript segment for virality potential."""
        if not segment.text.strip():
            return {'score': 0.0, 'reasons': [], 'features': {}}
        
        features = {}
        reasons = []
        score = 0.0
        
        text = segment.text.strip()
        text_lower = text.lower()
        
        # Basic linguistic features
        features['word_count'] = len(text.split())
        features['char_count'] = len(text)
        features['has_question'] = '?' in text
        features['has_exclamation'] = '!' in text
        features['is_caps'] = text.isupper()
        
        # Viral keywords and phrases
        viral_keywords = {
            'surprise': ['wow', 'whoa', 'no way', 'incredible', 'amazing', 'unbelievable'],
            'emotion': ['love', 'hate', 'excited', 'crazy', 'insane', 'epic'],
            'social': ['guys', 'everyone', 'people', 'friends', 'family'],
            'action': ['look', 'watch', 'see', 'check', 'try', 'do'],
            'emphasis': ['literally', 'actually', 'seriously', 'definitely', 'absolutely']
        }
        
        keyword_score = 0.0
        found_categories = []
        
        for category, keywords in viral_keywords.items():
            found = [kw for kw in keywords if kw in text_lower]
            if found:
                keyword_score += len(found) * 0.1
                found_categories.append(category)
                reasons.append(f"Contains {category} keywords: {', '.join(found)}")
        
        score += min(keyword_score, 0.3)
        features['viral_keywords'] = found_categories
        
        # Sentiment analysis
        if self.sentiment_pipeline:
            try:
                sentiment_result = self.sentiment_pipeline(text)[0]
                sentiment_label = sentiment_result['label'].lower()
                sentiment_score = sentiment_result['score']
                
                features['sentiment'] = sentiment_label
                features['sentiment_confidence'] = sentiment_score
                
                # Boost score for positive or very negative sentiment
                if sentiment_label in ['positive', 'joy'] and sentiment_score > 0.8:
                    score += 0.2
                    reasons.append(f"Strong positive sentiment ({sentiment_score:.2f})")
                elif sentiment_label in ['negative', 'anger'] and sentiment_score > 0.9:
                    score += 0.15  # Controversial content can be viral
                    reasons.append(f"Strong negative sentiment ({sentiment_score:.2f})")
                
            except Exception as e:
                console.print(f"[yellow]Sentiment analysis error: {e}[/yellow]")
        
        # Length-based scoring
        if 5 <= features['word_count'] <= 20:  # Sweet spot for viral clips
            score += 0.1
            reasons.append("Optimal length for viral content")
        
        # Question/exclamation bonus
        if features['has_question']:
            score += 0.1
            reasons.append("Contains engaging question")
        
        if features['has_exclamation']:
            score += 0.05
            reasons.append("Contains excitement markers")
        
        # Repetition and emphasis
        words = text_lower.split()
        if len(set(words)) < len(words) * 0.8:  # Some repetition
            score += 0.05
            reasons.append("Contains repetition for emphasis")
        
        return {
            'score': min(score, 1.0),
            'reasons': reasons,
            'features': features,
            'confidence': 0.8 if self.sentiment_pipeline else 0.5
        }
    
    def score_visual_features(self, objects: List[DetectedObject], 
                            emotions: List[EmotionDetection],
                            actions: List[ActionDetection]) -> Dict[str, Any]:
        """Score visual features for virality potential."""
        features = {}
        reasons = []
        score = 0.0
        
        # Object-based scoring
        interesting_objects = {
            'person': 0.1,
            'dog': 0.15,
            'cat': 0.15,
            'car': 0.05,
            'sports ball': 0.1,
            'bicycle': 0.05,
            'motorcycle': 0.08,
            'food': 0.1
        }
        
        object_counts = {}
        for obj in objects:
            if obj.class_name not in object_counts:
                object_counts[obj.class_name] = 0
            object_counts[obj.class_name] += 1
        
        features['object_counts'] = object_counts
        
        for obj_name, count in object_counts.items():
            if obj_name in interesting_objects:
                obj_score = interesting_objects[obj_name] * min(count, 3)  # Cap at 3
                score += obj_score
                if count > 1:
                    reasons.append(f"Multiple {obj_name}s detected ({count})")
                else:
                    reasons.append(f"{obj_name.title()} detected")
        
        # Emotion-based scoring
        emotion_weights = {
            'happy': 0.2,
            'surprise': 0.25,
            'excitement': 0.25,
            'joy': 0.2,
            'fear': 0.1,  # Can be viral if authentic
            'angry': 0.05,
            'sad': 0.05
        }
        
        emotion_counts = {}
        max_emotion_confidence = 0.0
        
        for emotion in emotions:
            emotion_name = emotion.emotion.lower()
            if emotion_name not in emotion_counts:
                emotion_counts[emotion_name] = []
            emotion_counts[emotion_name].append(emotion.confidence)
            max_emotion_confidence = max(max_emotion_confidence, emotion.confidence)
        
        features['emotion_counts'] = {k: len(v) for k, v in emotion_counts.items()}
        features['max_emotion_confidence'] = max_emotion_confidence
        
        for emotion_name, confidences in emotion_counts.items():
            if emotion_name in emotion_weights:
                avg_confidence = np.mean(confidences)
                emotion_score = emotion_weights[emotion_name] * avg_confidence * len(confidences)
                score += min(emotion_score, 0.3)
                
                if len(confidences) > 1:
                    reasons.append(f"Multiple {emotion_name} expressions detected")
                elif avg_confidence > 0.8:
                    reasons.append(f"Strong {emotion_name} expression ({avg_confidence:.2f})")
        
        # Action-based scoring
        action_weights = {
            'running': 0.15,
            'jumping': 0.2,
            'dancing': 0.25,
            'driving_fast': 0.1,
            'sports': 0.2
        }
        
        action_scores = []
        for action in actions:
            action_name = action.action.lower()
            for weighted_action, weight in action_weights.items():
                if weighted_action in action_name:
                    action_score = weight * action.confidence
                    action_scores.append(action_score)
                    reasons.append(f"Detected {action_name} action")
                    break
        
        if action_scores:
            score += min(sum(action_scores), 0.25)
            features['action_score'] = sum(action_scores)
        
        # Bonus for multiple people
        person_count = object_counts.get('person', 0)
        if person_count > 1:
            score += min(person_count * 0.05, 0.2)
            reasons.append(f"Multiple people in scene ({person_count})")
        
        return {
            'score': min(score, 1.0),
            'reasons': reasons,
            'features': features,
            'confidence': 0.7
        }
    
    def score_audio_features(self, transcript_segments: List[TranscriptSegment],
                           silence_segments: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Score audio features for virality potential."""
        features = {}
        reasons = []
        score = 0.0
        
        if not transcript_segments:
            return {'score': 0.0, 'reasons': [], 'features': {}, 'confidence': 0.0}
        
        # Speaking rate analysis
        total_speech_time = sum(seg.end - seg.start for seg in transcript_segments)
        total_words = sum(len(seg.text.split()) for seg in transcript_segments)
        
        if total_speech_time > 0:
            words_per_minute = (total_words / total_speech_time) * 60
            features['words_per_minute'] = words_per_minute
            
            # Optimal speaking rate for engagement
            if 140 <= words_per_minute <= 180:
                score += 0.15
                reasons.append(f"Optimal speaking rate ({words_per_minute:.0f} WPM)")
            elif words_per_minute > 200:  # Fast, energetic
                score += 0.1
                reasons.append(f"High-energy speaking rate ({words_per_minute:.0f} WPM)")
        
        # Silence analysis
        if silence_segments:
            avg_silence = np.mean([end - start for start, end in silence_segments])
            features['avg_silence_duration'] = avg_silence
            
            # Good use of pauses
            if 0.5 <= avg_silence <= 2.0:
                score += 0.05
                reasons.append("Good use of dramatic pauses")
        
        # Dialogue analysis
        speakers = set(seg.speaker for seg in transcript_segments if seg.speaker)
        features['speaker_count'] = len(speakers)
        
        if len(speakers) > 1:
            score += min(len(speakers) * 0.1, 0.2)
            reasons.append(f"Multi-speaker conversation ({len(speakers)} speakers)")
        
        # Text-based audio features
        all_text = " ".join(seg.text for seg in transcript_segments)
        
        # Excitement indicators
        excitement_markers = all_text.count('!') + all_text.count('?')
        if excitement_markers > 0:
            score += min(excitement_markers * 0.02, 0.1)
            reasons.append(f"High excitement in speech ({excitement_markers} markers)")
        
        # Laughter and vocal expressions
        vocal_expressions = ['haha', 'hehe', 'wow', 'whoa', 'oh my', 'oh no']
        expression_count = sum(all_text.lower().count(expr) for expr in vocal_expressions)
        if expression_count > 0:
            score += min(expression_count * 0.05, 0.15)
            reasons.append("Contains vocal expressions and reactions")
        
        return {
            'score': min(score, 1.0),
            'reasons': reasons,
            'features': features,
            'confidence': 0.6
        }
    
    def find_viral_moments(self, transcript_segments: List[TranscriptSegment],
                          visual_objects: List[DetectedObject],
                          emotions: List[EmotionDetection],
                          actions: List[ActionDetection],
                          silence_segments: List[Tuple[float, float]],
                          min_duration: float = 15.0,
                          max_duration: float = 60.0) -> List[ViralMoment]:
        """Find the best viral moments in the video."""
        
        if not transcript_segments:
            console.print("[yellow]No transcript segments provided[/yellow]")
            return []
        
        console.print("[blue]Analyzing video for viral moments[/blue]")
        
        # Create time windows for analysis
        video_duration = max(seg.end for seg in transcript_segments)
        window_size = 5.0  # 5-second windows
        windows = []
        
        for start_time in np.arange(0, video_duration, window_size):
            end_time = min(start_time + window_size, video_duration)
            windows.append((start_time, end_time))
        
        # Score each window
        window_scores = []
        
        for start_time, end_time in windows:
            # Get relevant data for this window
            window_transcript = [seg for seg in transcript_segments 
                               if seg.start >= start_time and seg.end <= end_time]
            window_objects = [obj for obj in visual_objects 
                            if start_time <= obj.timestamp <= end_time]
            window_emotions = [em for em in emotions 
                             if start_time <= em.timestamp <= end_time]
            window_actions = [act for act in actions 
                            if not (act.end_time < start_time or act.start_time > end_time)]
            window_silence = [(s, e) for s, e in silence_segments 
                            if not (e < start_time or s > end_time)]
            
            # Score different aspects
            transcript_score = 0.0
            if window_transcript:
                transcript_scores = [self.score_transcript_segment(seg) for seg in window_transcript]
                transcript_score = np.mean([ts['score'] for ts in transcript_scores])
            
            visual_analysis = self.score_visual_features(window_objects, window_emotions, window_actions)
            visual_score = visual_analysis['score']
            
            audio_analysis = self.score_audio_features(window_transcript, window_silence)
            audio_score = audio_analysis['score']
            
            # Weighted combination
            total_score = (
                transcript_score * config.EMOTION_WEIGHT +
                visual_score * config.VISUAL_WEIGHT +
                audio_score * config.AUDIO_WEIGHT +
                (len(window_objects) > 0) * config.ACTION_WEIGHT * 0.5
            )
            
            window_scores.append({
                'start_time': start_time,
                'end_time': end_time,
                'score': total_score,
                'transcript_score': transcript_score,
                'visual_score': visual_score,
                'audio_score': audio_score,
                'transcript_segments': window_transcript,
                'visual_analysis': visual_analysis,
                'audio_analysis': audio_analysis
            })
        
        # Find high-scoring continuous regions
        viral_moments = []
        if window_scores:
            # Sort by score
            window_scores.sort(key=lambda x: x['score'], reverse=True)
            
            # Group high-scoring adjacent windows
            used_windows = set()
            
            for window in window_scores:
                if window['score'] < 0.3:  # Minimum threshold
                    break
                
                if window['start_time'] in used_windows:
                    continue
                
                # Extend moment by including adjacent high-scoring windows
                moment_start = window['start_time']
                moment_end = window['end_time']
                moment_score = window['score']
                moment_windows = [window]
                
                # Look for adjacent windows
                for other_window in window_scores:
                    if (other_window['start_time'] in used_windows or 
                        other_window['score'] < 0.2):
                        continue
                    
                    # Check if adjacent
                    if (abs(other_window['end_time'] - moment_start) < 1.0 or
                        abs(other_window['start_time'] - moment_end) < 1.0):
                        
                        moment_start = min(moment_start, other_window['start_time'])
                        moment_end = max(moment_end, other_window['end_time'])
                        moment_score = max(moment_score, other_window['score'])
                        moment_windows.append(other_window)
                        used_windows.add(other_window['start_time'])
                
                # Check duration constraints
                duration = moment_end - moment_start
                if min_duration <= duration <= max_duration:
                    # Compile reasons
                    all_reasons = []
                    for mw in moment_windows:
                        all_reasons.extend(mw['visual_analysis']['reasons'])
                        all_reasons.extend(mw['audio_analysis']['reasons'])
                    
                    # Remove duplicates
                    unique_reasons = list(set(all_reasons))
                    
                    viral_moment = ViralMoment(
                        start_time=moment_start,
                        end_time=moment_end,
                        duration=duration,
                        virality_score=moment_score,
                        reasons=unique_reasons,
                        transcript_segments=[seg for mw in moment_windows 
                                           for seg in mw['transcript_segments']],
                        visual_features=moment_windows[0]['visual_analysis']['features'],
                        audio_features=moment_windows[0]['audio_analysis']['features'],
                        confidence=0.8
                    )
                    
                    viral_moments.append(viral_moment)
                    used_windows.add(window['start_time'])
        
        # Sort by score and return top moments
        viral_moments.sort(key=lambda x: x.virality_score, reverse=True)
        
        console.print(f"[green]Found {len(viral_moments)} potential viral moments[/green]")
        
        return viral_moments[:5]  # Return top 5 moments
    
    def generate_viral_summary(self, viral_moment: ViralMoment) -> str:
        """Generate a summary description of why this moment is viral."""
        summary_parts = []
        
        # Add duration info
        summary_parts.append(f"🎬 {viral_moment.duration:.1f}s clip ({viral_moment.start_time:.1f}s - {viral_moment.end_time:.1f}s)")
        
        # Add score
        summary_parts.append(f"⭐ Virality Score: {viral_moment.virality_score:.2f}/1.0")
        
        # Add key reasons
        if viral_moment.reasons:
            summary_parts.append("🔥 Key Factors:")
            for reason in viral_moment.reasons[:5]:  # Top 5 reasons
                summary_parts.append(f"  • {reason}")
        
        # Add transcript preview
        if viral_moment.transcript_segments:
            all_text = " ".join(seg.text for seg in viral_moment.transcript_segments)
            if len(all_text) > 100:
                all_text = all_text[:100] + "..."
            summary_parts.append(f"💬 Content: \"{all_text}\"")
        
        return "\n".join(summary_parts)

def find_viral_moments(transcript_segments: List[TranscriptSegment],
                      visual_objects: List[DetectedObject],
                      emotions: List[EmotionDetection],
                      actions: List[ActionDetection]) -> List[ViralMoment]:
    """Convenience function to find viral moments."""
    scorer = ViralityScorer()
    return scorer.find_viral_moments(transcript_segments, visual_objects, emotions, actions, [])