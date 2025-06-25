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
from .visual_analysis import DetectedObject, EmotionDetection, ActionDetection, SceneClassification

console = Console()

@dataclass
class EmotionArc:
    """Represents an emotional narrative arc."""
    start_time: float
    end_time: float
    arc_type: str  # tension_buildup, peak_moment, resolution, etc.
    emotion_trajectory: List[Tuple[float, str, float]]  # (timestamp, emotion, intensity)
    narrative_strength: float
    speaker_changes: int

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
    emotion_arc: Optional[EmotionArc] = None

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
            
            # Load sentiment analysis pipeline from local models
            sentiment_model_path = config.MODELS_DIR / "sentiment-model"
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=str(sentiment_model_path),
                device=0 if "cuda" in self.device else -1
            )
            
            # Load sentence transformer for semantic similarity from local models
            console.print("[blue]Loading sentence transformer[/blue]")
            sentence_transformer_path = config.MODELS_DIR / "sentence-transformer"
            self.sentence_transformer = SentenceTransformer(
                str(sentence_transformer_path),
                device=self.device
            )
            
            # Load main LLM for text generation from local models
            console.print(f"[blue]Loading main LLM: {self.model_name}[/blue]")
            dialogpt_model_path = config.MODELS_DIR / "dialogpt-model"
            self.tokenizer = AutoTokenizer.from_pretrained(str(dialogpt_model_path))
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                str(dialogpt_model_path),
                quantization_config=quantization_config,
                device_map="auto" if "cuda" in self.device else None,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                low_cpu_mem_usage=True
            )
            
            console.print("[green]All virality models loaded successfully[/green]")
            
        except Exception as e:
            console.print(f"[red]Error loading virality models: {e}[/red]")
            console.print("[yellow]Falling back to simplified scoring[/yellow]")
    
    def analyze_emotion_progression(self, transcript_segments: List[TranscriptSegment],
                                  emotions: List[EmotionDetection],
                                  scene_classifications: List[SceneClassification] = None) -> List[EmotionArc]:
        """Analyze emotional progression to identify narrative arcs."""
        if not transcript_segments and not emotions:
            return []
        
        # Combine text and visual emotions into timeline
        emotion_timeline = []
        
        # Add text-based emotions
        for segment in transcript_segments:
            text_emotion = self._extract_text_emotion(segment)
            if text_emotion:
                emotion_timeline.append({
                    'timestamp': (segment.start + segment.end) / 2,
                    'emotion': text_emotion['emotion'],
                    'intensity': text_emotion['intensity'],
                    'source': 'text',
                    'speaker': segment.speaker
                })
        
        # Add visual emotions
        for emotion in emotions:
            emotion_timeline.append({
                'timestamp': emotion.timestamp,
                'emotion': emotion.emotion,
                'intensity': emotion.confidence,
                'source': 'visual',
                'speaker': None
            })
        
        # Sort by timestamp
        emotion_timeline.sort(key=lambda x: x['timestamp'])
        
        if not emotion_timeline:
            return []
        
        # Identify emotion arcs
        arcs = self._identify_emotion_arcs(emotion_timeline)
        
        # Enhance arcs with scene context
        if scene_classifications:
            arcs = self._enhance_arcs_with_scene_context(arcs, scene_classifications)
        
        return arcs
    
    def _extract_text_emotion(self, segment: TranscriptSegment) -> Optional[Dict[str, Any]]:
        """Extract emotion from text segment."""
        text = segment.text.lower()
        
        # Enhanced emotion mapping
        emotion_patterns = {
            'excitement': {
                'keywords': ['amazing', 'incredible', 'wow', 'awesome', 'fantastic', 'unbelievable'],
                'intensity_multiplier': 1.2
            },
            'surprise': {
                'keywords': ['whoa', 'what', 'no way', 'seriously', 'really', 'oh my god'],
                'intensity_multiplier': 1.1
            },
            'joy': {
                'keywords': ['happy', 'love', 'great', 'wonderful', 'perfect', 'haha'],
                'intensity_multiplier': 1.0
            },
            'tension': {
                'keywords': ['but', 'however', 'wait', 'problem', 'issue', 'difficult'],
                'intensity_multiplier': 0.9
            },
            'anticipation': {
                'keywords': ['going to', 'will', 'about to', 'ready', 'prepare', 'next'],
                'intensity_multiplier': 0.8
            },
            'resolution': {
                'keywords': ['finally', 'solved', 'done', 'finished', 'complete', 'success'],
                'intensity_multiplier': 1.0
            }
        }
        
        best_emotion = None
        best_score = 0
        
        for emotion, config in emotion_patterns.items():
            score = 0
            for keyword in config['keywords']:
                if keyword in text:
                    score += config['intensity_multiplier']
            
            # Boost for emphasis markers
            if '!' in segment.text:
                score *= 1.2
            if '?' in segment.text and emotion in ['surprise', 'anticipation']:
                score *= 1.1
            
            if score > best_score:
                best_score = score
                best_emotion = emotion
        
        if best_emotion and best_score > 0.5:
            return {
                'emotion': best_emotion,
                'intensity': min(best_score, 1.0)
            }
        
        return None
    
    def _identify_emotion_arcs(self, emotion_timeline: List[Dict]) -> List[EmotionArc]:
        """Identify narrative emotion arcs from timeline."""
        if len(emotion_timeline) < 3:
            return []
        
        arcs = []
        window_size = 5  # seconds
        min_arc_duration = 10  # seconds
        
        i = 0
        while i < len(emotion_timeline) - 2:
            arc_start = emotion_timeline[i]['timestamp']
            arc_emotions = [emotion_timeline[i]]
            
            # Collect emotions within time window
            j = i + 1
            while (j < len(emotion_timeline) and 
                   emotion_timeline[j]['timestamp'] - arc_start < window_size * 3):
                arc_emotions.append(emotion_timeline[j])
                j += 1
            
            if len(arc_emotions) < 3:
                i += 1
                continue
            
            arc_end = arc_emotions[-1]['timestamp']
            if arc_end - arc_start < min_arc_duration:
                i += 1
                continue
            
            # Analyze arc pattern
            arc_type, strength = self._classify_emotion_arc(arc_emotions)
            
            if strength > 0.3:  # Minimum strength threshold
                # Count speaker changes
                speakers = set(e.get('speaker') for e in arc_emotions if e.get('speaker'))
                speaker_changes = len(speakers) - 1 if len(speakers) > 1 else 0
                
                # Create emotion trajectory
                trajectory = [(e['timestamp'], e['emotion'], e['intensity']) for e in arc_emotions]
                
                arc = EmotionArc(
                    start_time=arc_start,
                    end_time=arc_end,
                    arc_type=arc_type,
                    emotion_trajectory=trajectory,
                    narrative_strength=strength,
                    speaker_changes=speaker_changes
                )
                arcs.append(arc)
            
            # Move forward, but allow overlap
            i += max(1, len(arc_emotions) // 2)
        
        return arcs
    
    def _classify_emotion_arc(self, arc_emotions: List[Dict]) -> Tuple[str, float]:
        """Classify the type and strength of an emotion arc."""
        if len(arc_emotions) < 3:
            return "minimal", 0.1
        
        # Extract emotion sequence
        emotions = [e['emotion'] for e in arc_emotions]
        intensities = [e['intensity'] for e in arc_emotions]
        
        # Define arc patterns
        tension_buildup_patterns = [
            ['anticipation', 'tension', 'excitement'],
            ['joy', 'tension', 'surprise'],
            ['tension', 'tension', 'excitement']
        ]
        
        peak_resolution_patterns = [
            ['excitement', 'surprise', 'joy'],
            ['tension', 'excitement', 'resolution'],
            ['surprise', 'excitement', 'resolution']
        ]
        
        # Check for tension-buildup arc
        if self._matches_pattern_sequence(emotions, tension_buildup_patterns):
            strength = np.mean(intensities) * 1.2  # Bonus for buildup
            return "tension_buildup", min(strength, 1.0)
        
        # Check for peak-resolution arc
        if self._matches_pattern_sequence(emotions, peak_resolution_patterns):
            strength = np.mean(intensities) * 1.1
            return "peak_resolution", min(strength, 1.0)
        
        # Check for sustained excitement
        if emotions.count('excitement') >= len(emotions) * 0.6:
            strength = np.mean(intensities)
            return "sustained_excitement", min(strength, 1.0)
        
        # Check for dramatic contrast
        intensity_range = max(intensities) - min(intensities)
        if intensity_range > 0.5:
            strength = intensity_range
            return "dramatic_contrast", min(strength, 1.0)
        
        # Default to general arc
        strength = np.mean(intensities) * 0.8
        return "general_progression", min(strength, 1.0)
    
    def _matches_pattern_sequence(self, emotions: List[str], patterns: List[List[str]]) -> bool:
        """Check if emotion sequence matches any of the given patterns."""
        for pattern in patterns:
            if len(emotions) >= len(pattern):
                # Check for pattern match with some flexibility
                matches = 0
                for i, pattern_emotion in enumerate(pattern):
                    if i < len(emotions) and emotions[i] == pattern_emotion:
                        matches += 1
                    elif i < len(emotions) and pattern_emotion in emotions[max(0, i-1):i+2]:
                        matches += 0.5  # Partial match for nearby emotions
                
                if matches >= len(pattern) * 0.7:  # 70% match threshold
                    return True
        return False
    
    def _enhance_arcs_with_scene_context(self, arcs: List[EmotionArc], 
                                       scene_classifications: List[SceneClassification]) -> List[EmotionArc]:
        """Enhance emotion arcs with scene context information."""
        enhanced_arcs = []
        
        for arc in arcs:
            # Find relevant scenes for this arc
            relevant_scenes = []
            for scene in scene_classifications:
                if arc.start_time <= scene.timestamp <= arc.end_time:
                    relevant_scenes.append(scene)
            
            # Adjust narrative strength based on scene context
            strength_modifier = 1.0
            
            if relevant_scenes:
                # Social scenes enhance emotion arcs
                social_scenes = [s for s in relevant_scenes if s.scene_type == 'social']
                if social_scenes:
                    strength_modifier *= 1.2
                
                # Action/sports scenes enhance excitement arcs
                action_scenes = [s for s in relevant_scenes if s.scene_type in ['sports', 'action']]
                if action_scenes and arc.arc_type in ['excitement', 'sustained_excitement']:
                    strength_modifier *= 1.15
                
                # Presentation scenes enhance tension-buildup arcs
                presentation_scenes = [s for s in relevant_scenes if s.scene_type == 'presentation']
                if presentation_scenes and arc.arc_type == 'tension_buildup':
                    strength_modifier *= 1.1
            
            # Create enhanced arc
            enhanced_arc = EmotionArc(
                start_time=arc.start_time,
                end_time=arc.end_time,
                arc_type=arc.arc_type,
                emotion_trajectory=arc.emotion_trajectory,
                narrative_strength=min(arc.narrative_strength * strength_modifier, 1.0),
                speaker_changes=arc.speaker_changes
            )
            enhanced_arcs.append(enhanced_arc)
        
        return enhanced_arcs
    
    def get_content_aware_weights(self, transcript_segments: List[TranscriptSegment],
                                scene_classifications: List[SceneClassification],
                                emotions: List[EmotionDetection],
                                actions: List[ActionDetection]) -> Dict[str, float]:
        """Calculate dynamic scoring weights based on content type and context."""
        # Default weights from config
        weights = {
            'emotion': config.EMOTION_WEIGHT,
            'visual': config.VISUAL_WEIGHT,
            'audio': config.AUDIO_WEIGHT,
            'action': config.ACTION_WEIGHT
        }
        
        # Analyze content characteristics
        content_type = self._determine_primary_content_type(
            transcript_segments, scene_classifications, emotions, actions
        )
        
        # Adjust weights based on content type
        if content_type == 'educational':
            # Educational content: emphasize clear speech and visual aids
            weights['audio'] *= 1.3
            weights['visual'] *= 1.1
            weights['emotion'] *= 0.9
            weights['action'] *= 0.8
            
        elif content_type == 'entertainment':
            # Entertainment: emphasize emotions and actions
            weights['emotion'] *= 1.4
            weights['action'] *= 1.3
            weights['visual'] *= 1.1
            weights['audio'] *= 0.9
            
        elif content_type == 'conversation':
            # Conversations: emphasize speaker dynamics and emotions
            weights['audio'] *= 1.4
            weights['emotion'] *= 1.2
            weights['visual'] *= 0.8
            weights['action'] *= 0.8
            
        elif content_type == 'action_sports':
            # Action/Sports: emphasize visual and action elements
            weights['action'] *= 1.5
            weights['visual'] *= 1.3
            weights['emotion'] *= 1.1
            weights['audio'] *= 0.7
            
        elif content_type == 'cooking_tutorial':
            # Cooking: balance visual demonstration with clear instruction
            weights['visual'] *= 1.3
            weights['action'] *= 1.2
            weights['audio'] *= 1.1
            weights['emotion'] *= 0.9
            
        elif content_type == 'technology':
            # Tech content: emphasize clear explanation and visual demonstration
            weights['audio'] *= 1.2
            weights['visual'] *= 1.2
            weights['action'] *= 1.0
            weights['emotion'] *= 0.9
            
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _determine_primary_content_type(self, transcript_segments: List[TranscriptSegment],
                                      scene_classifications: List[SceneClassification],
                                      emotions: List[EmotionDetection],
                                      actions: List[ActionDetection]) -> str:
        """Determine the primary content type based on analysis."""
        type_scores = {}
        
        # Analyze transcript for content indicators
        all_text = " ".join(seg.text for seg in transcript_segments).lower()
        
        # Educational indicators
        educational_keywords = [
            'learn', 'teach', 'explain', 'understand', 'tutorial', 'how to',
            'guide', 'step', 'method', 'technique', 'lesson', 'course'
        ]
        educational_score = sum(all_text.count(kw) for kw in educational_keywords) / max(len(all_text.split()), 1)
        type_scores['educational'] = educational_score
        
        # Entertainment indicators
        entertainment_keywords = [
            'funny', 'hilarious', 'comedy', 'joke', 'laugh', 'fun', 'entertaining',
            'awesome', 'amazing', 'incredible', 'wow', 'crazy'
        ]
        entertainment_score = sum(all_text.count(kw) for kw in entertainment_keywords) / max(len(all_text.split()), 1)
        type_scores['entertainment'] = entertainment_score
        
        # Conversation indicators
        conversation_indicators = 0
        if transcript_segments:
            # Multiple speakers
            speakers = set(seg.speaker for seg in transcript_segments if seg.speaker)
            if len(speakers) > 1:
                conversation_indicators += len(speakers) * 0.1
            
            # Question/answer patterns
            questions = sum('?' in seg.text for seg in transcript_segments)
            conversation_indicators += questions * 0.05
            
        type_scores['conversation'] = conversation_indicators
        
        # Analyze scene classifications
        if scene_classifications:
            scene_types = [sc.scene_type for sc in scene_classifications]
            
            # Sports/Action content
            action_scenes = sum(1 for st in scene_types if st in ['sports', 'action'])
            type_scores['action_sports'] = action_scenes / len(scene_types)
            
            # Cooking content
            cooking_scenes = sum(1 for st in scene_types if st == 'cooking')
            type_scores['cooking_tutorial'] = cooking_scenes / len(scene_types)
            
            # Technology content
            tech_scenes = sum(1 for st in scene_types if st == 'technology')
            type_scores['technology'] = tech_scenes / len(scene_types)
        
        # Analyze actions for content type
        if actions:
            action_types = [act.action for act in actions]
            
            # Sports actions boost action_sports score
            sports_actions = sum(1 for at in action_types if 'running' in at or 'moving_fast' in at)
            if 'action_sports' in type_scores:
                type_scores['action_sports'] += sports_actions / len(action_types) * 0.5
            else:
                type_scores['action_sports'] = sports_actions / len(action_types) * 0.5
        
        # Return the content type with highest score
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            if best_type[1] > 0.1:  # Minimum threshold
                return best_type[0]
        
        # Default to entertainment if no clear type detected
        return 'entertainment'
    
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
                          scene_classifications: List[SceneClassification] = None,
                          min_duration: float = 15.0,
                          max_duration: float = 60.0) -> List[ViralMoment]:
        """Find the best viral moments in the video."""
        
        if not transcript_segments:
            console.print("[yellow]No transcript segments provided[/yellow]")
            return []
        
        console.print("[blue]Analyzing video for viral moments[/blue]")
        
        # Get content-aware weights
        adaptive_weights = self.get_content_aware_weights(
            transcript_segments, scene_classifications or [], emotions, actions
        )
        console.print(f"[cyan]Using adaptive weights: {adaptive_weights}[/cyan]")
        
        # Analyze emotion progression for narrative arcs
        emotion_arcs = self.analyze_emotion_progression(
            transcript_segments, emotions, scene_classifications
        )
        
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
            
            # Content-aware weighted combination
            total_score = (
                transcript_score * adaptive_weights['emotion'] +
                visual_score * adaptive_weights['visual'] +
                audio_score * adaptive_weights['audio'] +
                (len(window_objects) > 0) * adaptive_weights['action'] * 0.5
            )
            
            # Bonus for emotion arc alignment
            emotion_arc_bonus = 0.0
            for arc in emotion_arcs:
                if arc.start_time <= (start_time + end_time) / 2 <= arc.end_time:
                    # Bonus based on arc type and strength
                    if arc.arc_type in ['tension_buildup', 'peak_resolution']:
                        emotion_arc_bonus += arc.narrative_strength * 0.2
                    elif arc.arc_type == 'dramatic_contrast':
                        emotion_arc_bonus += arc.narrative_strength * 0.15
                    break
            
            total_score += emotion_arc_bonus
            
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
                    
                    # Find relevant emotion arc for this moment
                    relevant_arc = None
                    moment_mid = (moment_start + moment_end) / 2
                    for arc in emotion_arcs:
                        if arc.start_time <= moment_mid <= arc.end_time:
                            relevant_arc = arc
                            break
                    
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
                        confidence=0.8,
                        emotion_arc=relevant_arc
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