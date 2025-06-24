"""Visual analysis pipeline using YOLO, emotion detection, and action recognition."""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from ultralytics import YOLO
from transformers import pipeline, CLIPProcessor, CLIPModel
from PIL import Image
import face_recognition
from rich.console import Console
from rich.progress import Progress, track

from .config import config

console = Console()

@dataclass
class DetectedObject:
    """Represents a detected object in a frame."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    timestamp: float

@dataclass
class EmotionDetection:
    """Represents detected emotion in a frame."""
    emotion: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    timestamp: float = 0.0

@dataclass
class ActionDetection:
    """Represents detected action in a video segment."""
    action: str
    confidence: float
    start_time: float
    end_time: float

class VisualAnalyzer:
    """Comprehensive visual analysis pipeline for video content."""
    
    def __init__(self, device: str = None):
        self.device = device or self._get_best_device()
        self.yolo_model = None
        self.emotion_pipeline = None
        self.clip_model = None
        self.clip_processor = None
        
        # Set up GPU memory management for ROCm
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
        """Load all visual analysis models."""
        try:
            # Load YOLO for object detection
            console.print(f"[blue]Loading YOLO model: {config.YOLO_MODEL}[/blue]")
            self.yolo_model = YOLO(config.YOLO_MODEL)
            
            # Skip emotion detection for now - use simple face detection instead
            console.print("[blue]Emotion detection disabled - using face detection only[/blue]")
            self.emotion_pipeline = None
            
            # Load CLIP for semantic understanding from local models
            console.print("[blue]Loading CLIP model[/blue]")
            clip_model_path = config.MODELS_DIR / "clip-model"
            self.clip_model = CLIPModel.from_pretrained(str(clip_model_path))
            self.clip_processor = CLIPProcessor.from_pretrained(str(clip_model_path))
            
            if "cuda" in self.device:
                self.clip_model = self.clip_model.to(self.device)
            
            console.print("[green]All visual models loaded successfully[/green]")
            
        except Exception as e:
            console.print(f"[red]Error loading visual models: {e}[/red]")
            console.print("[yellow]Some features may not be available[/yellow]")
    
    def analyze_frames(self, frame_paths: List[Path], fps: float = 1.0) -> Dict[str, Any]:
        """Analyze a sequence of frames for objects, emotions, and actions."""
        results = {
            'objects': [],
            'emotions': [],
            'actions': [],
            'scene_changes': [],
            'highlights': []
        }
        
        if not frame_paths:
            return results
        
        console.print(f"[blue]Analyzing {len(frame_paths)} frames[/blue]")
        
        prev_frame_features = None
        scene_change_threshold = 0.3
        
        for i, frame_path in enumerate(frame_paths):
            timestamp = i / fps
            
            try:
                # Load frame
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    continue
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Object detection
                objects = self._detect_objects(frame_rgb, timestamp)
                results['objects'].extend(objects)
                
                # Emotion detection
                emotions = self._detect_emotions(frame_rgb, timestamp)
                results['emotions'].extend(emotions)
                
                # Scene change detection
                current_features = self._extract_frame_features(frame_rgb)
                if prev_frame_features is not None:
                    similarity = self._calculate_frame_similarity(prev_frame_features, current_features)
                    if similarity < scene_change_threshold:
                        results['scene_changes'].append({
                            'timestamp': timestamp,
                            'similarity': similarity
                        })
                
                prev_frame_features = current_features
                
                # Detect visual highlights
                highlight_score = self._calculate_visual_highlight_score(objects, emotions, frame_rgb)
                if highlight_score > 0.7:
                    results['highlights'].append({
                        'timestamp': timestamp,
                        'score': highlight_score,
                        'type': 'visual'
                    })
                
            except Exception as e:
                console.print(f"[yellow]Error processing frame {frame_path}: {e}[/yellow]")
                continue
        
        # Detect actions from object sequences
        results['actions'] = self._detect_actions_from_objects(results['objects'], fps)
        
        console.print(f"[green]Visual analysis completed: {len(results['objects'])} objects, "
                     f"{len(results['emotions'])} emotions, {len(results['actions'])} actions[/green]")
        
        return results
    
    def _detect_objects(self, frame: np.ndarray, timestamp: float) -> List[DetectedObject]:
        """Detect objects in a single frame using YOLO."""
        if not self.yolo_model:
            return []
        
        try:
            results = self.yolo_model(frame, conf=config.YOLO_CONFIDENCE, verbose=False)
            objects = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = self.yolo_model.names[cls]
                        
                        objects.append(DetectedObject(
                            class_name=class_name,
                            confidence=conf,
                            bbox=(x1, y1, x2, y2),
                            timestamp=timestamp
                        ))
            
            return objects
            
        except Exception as e:
            console.print(f"[yellow]Object detection error: {e}[/yellow]")
            return []
    
    def _detect_emotions(self, frame: np.ndarray, timestamp: float) -> List[EmotionDetection]:
        """Detect faces in the frame (emotion detection disabled for now)."""
        try:
            # Just detect faces and assign neutral emotion for now
            face_locations = face_recognition.face_locations(frame)
            emotions = []
            
            for face_location in face_locations:
                top, right, bottom, left = face_location
                # Just add faces as "neutral" emotions for scoring purposes
                emotions.append(EmotionDetection(
                    emotion="neutral",
                    confidence=0.8,
                    bbox=(left, top, right, bottom),
                    timestamp=timestamp
                ))
            
            return emotions
            
        except Exception as e:
            console.print(f"[yellow]Face detection error: {e}[/yellow]")
            return []
    
    def _extract_frame_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract features from frame for scene change detection."""
        try:
            # Simple histogram-based features
            hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
            hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])
            
            features = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
            return features / np.linalg.norm(features)  # Normalize
            
        except Exception as e:
            console.print(f"[yellow]Feature extraction error: {e}[/yellow]")
            return np.zeros(768)  # Return zero vector on error
    
    def _calculate_frame_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate similarity between two frame feature vectors."""
        try:
            return np.dot(features1, features2)
        except Exception:
            return 0.0
    
    def _calculate_visual_highlight_score(self, objects: List[DetectedObject], 
                                        emotions: List[EmotionDetection], 
                                        frame: np.ndarray) -> float:
        """Calculate a highlight score based on visual content."""
        score = 0.0
        
        # Score based on number of detected objects
        score += min(len(objects) * 0.1, 0.3)
        
        # Score based on emotions
        positive_emotions = ['happy', 'surprise', 'excitement']
        for emotion in emotions:
            if emotion.emotion.lower() in positive_emotions:
                score += emotion.confidence * 0.2
        
        # Score based on motion/activity (simplified using edge detection)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        score += min(edge_density * 2, 0.3)
        
        # Score based on interesting objects
        interesting_objects = ['person', 'car', 'dog', 'cat', 'sports ball', 'bicycle']
        for obj in objects:
            if obj.class_name in interesting_objects:
                score += obj.confidence * 0.15
        
        return min(score, 1.0)
    
    def _detect_actions_from_objects(self, objects: List[DetectedObject], fps: float) -> List[ActionDetection]:
        """Detect actions based on object movement patterns."""
        actions = []
        
        if not objects:
            return actions
        
        # Group objects by class and track movement
        object_tracks = {}
        for obj in objects:
            if obj.class_name not in object_tracks:
                object_tracks[obj.class_name] = []
            object_tracks[obj.class_name].append(obj)
        
        # Analyze movement patterns
        for class_name, track in object_tracks.items():
            if len(track) < 3:  # Need minimum frames for action detection
                continue
            
            # Sort by timestamp
            track.sort(key=lambda x: x.timestamp)
            
            # Calculate movement
            movements = []
            for i in range(1, len(track)):
                prev_obj = track[i-1]
                curr_obj = track[i]
                
                # Calculate center movement
                prev_center = ((prev_obj.bbox[0] + prev_obj.bbox[2]) / 2,
                              (prev_obj.bbox[1] + prev_obj.bbox[3]) / 2)
                curr_center = ((curr_obj.bbox[0] + curr_obj.bbox[2]) / 2,
                              (curr_obj.bbox[1] + curr_obj.bbox[3]) / 2)
                
                distance = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                                 (curr_center[1] - prev_center[1])**2)
                time_diff = curr_obj.timestamp - prev_obj.timestamp
                
                if time_diff > 0:
                    speed = distance / time_diff
                    movements.append(speed)
            
            # Classify actions based on movement patterns
            if movements:
                avg_speed = np.mean(movements)
                max_speed = np.max(movements)
                
                if class_name == 'person':
                    if avg_speed > 50:  # Fast movement
                        action = 'running' if max_speed > 100 else 'walking_fast'
                    elif avg_speed > 20:
                        action = 'walking'
                    else:
                        action = 'standing'
                elif class_name == 'car':
                    if avg_speed > 80:
                        action = 'driving_fast'
                    elif avg_speed > 30:
                        action = 'driving'
                    else:
                        action = 'parked'
                else:
                    if avg_speed > 40:
                        action = f'{class_name}_moving_fast'
                    elif avg_speed > 10:
                        action = f'{class_name}_moving'
                    else:
                        action = f'{class_name}_static'
                
                # Calculate confidence based on consistency
                confidence = min(1.0, 1.0 / (1.0 + np.std(movements) / max(avg_speed, 1)))
                
                actions.append(ActionDetection(
                    action=action,
                    confidence=confidence,
                    start_time=track[0].timestamp,
                    end_time=track[-1].timestamp
                ))
        
        return actions
    
    def analyze_single_frame(self, frame_path: Path) -> Dict[str, Any]:
        """Analyze a single frame for all visual features."""
        try:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                return {}
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Object detection
            objects = self._detect_objects(frame_rgb, 0.0)
            
            # Emotion detection
            emotions = self._detect_emotions(frame_rgb, 0.0)
            
            # Calculate visual appeal score
            highlight_score = self._calculate_visual_highlight_score(objects, emotions, frame_rgb)
            
            return {
                'objects': objects,
                'emotions': emotions,
                'highlight_score': highlight_score,
                'frame_path': str(frame_path)
            }
            
        except Exception as e:
            console.print(f"[red]Error analyzing frame {frame_path}: {e}[/red]")
            return {}
    
    def find_best_thumbnail_frame(self, frame_paths: List[Path]) -> Optional[Path]:
        """Find the best frame for thumbnail based on visual appeal."""
        if not frame_paths:
            return None
        
        best_frame = None
        best_score = 0.0
        
        console.print("[blue]Finding best thumbnail frame[/blue]")
        
        for frame_path in frame_paths[:min(30, len(frame_paths))]:
            analysis = self.analyze_single_frame(frame_path)
            score = analysis.get('highlight_score', 0.0)
            
            # Bonus for faces
            emotions = analysis.get('emotions', [])
            if emotions:
                positive_emotions = sum(1 for e in emotions if e.emotion.lower() in ['happy', 'surprise'])
                score += positive_emotions * 0.2
            
            if score > best_score:
                best_score = score
                best_frame = frame_path
        
        console.print(f"[green]Best thumbnail frame: {best_frame} (score: {best_score:.2f})[/green]")
        return best_frame
    
    def detect_viral_moments(self, analysis_results: Dict[str, Any], 
                           duration_threshold: float = 5.0) -> List[Dict[str, Any]]:
        """Detect potential viral moments based on visual analysis."""
        viral_moments = []
        
        # Combine all highlights
        all_highlights = []
        
        # Add visual highlights
        for highlight in analysis_results.get('highlights', []):
            all_highlights.append({
                'timestamp': highlight['timestamp'],
                'score': highlight['score'],
                'type': 'visual',
                'data': highlight
            })
        
        # Add emotion-based highlights
        emotions = analysis_results.get('emotions', [])
        for emotion in emotions:
            if (emotion.emotion.lower() in ['happy', 'surprise', 'excitement'] and 
                emotion.confidence > 0.8):
                all_highlights.append({
                    'timestamp': emotion.timestamp,
                    'score': emotion.confidence,
                    'type': 'emotion',
                    'data': emotion
                })
        
        # Sort by timestamp
        all_highlights.sort(key=lambda x: x['timestamp'])
        
        # Group nearby highlights into moments
        if all_highlights:
            current_moment = {
                'start_time': all_highlights[0]['timestamp'],
                'end_time': all_highlights[0]['timestamp'] + 1.0,
                'score': all_highlights[0]['score'],
                'highlights': [all_highlights[0]]
            }
            
            for highlight in all_highlights[1:]:
                if highlight['timestamp'] - current_moment['end_time'] < duration_threshold:
                    # Extend current moment
                    current_moment['end_time'] = highlight['timestamp'] + 1.0
                    current_moment['score'] = max(current_moment['score'], highlight['score'])
                    current_moment['highlights'].append(highlight)
                else:
                    # Save current moment and start new one
                    if len(current_moment['highlights']) >= 2:  # Minimum highlights for viral moment
                        viral_moments.append(current_moment)
                    
                    current_moment = {
                        'start_time': highlight['timestamp'],
                        'end_time': highlight['timestamp'] + 1.0,
                        'score': highlight['score'],
                        'highlights': [highlight]
                    }
            
            # Add the last moment
            if len(current_moment['highlights']) >= 2:
                viral_moments.append(current_moment)
        
        # Sort by score
        viral_moments.sort(key=lambda x: x['score'], reverse=True)
        
        console.print(f"[green]Detected {len(viral_moments)} potential viral moments[/green]")
        return viral_moments

def analyze_video_frames(frame_paths: List[Path], fps: float = 1.0) -> Dict[str, Any]:
    """Convenience function to analyze video frames."""
    analyzer = VisualAnalyzer()
    return analyzer.analyze_frames(frame_paths, fps)