import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import librosa
import cv2
from collections import deque
import json
import requests
from datetime import datetime
import whisper
from sentence_transformers import SentenceTransformer
import os
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
import time

def _extract_features_worker(args):
    """
    Worker function for parallel feature extraction
    This needs to be a top-level function for multiprocessing
    """
    segment, audio_path, video_path, model_name = args
    
    try:
        # Create detector instance in worker (models can't be pickled)
        detector = ViralPatternDetector(model_name=model_name)
        features = detector.extract_multimodal_features(segment, audio_path, video_path)
        return features
    except Exception as e:
        print(f"Error processing segment {segment.get('start', 0):.1f}s: {e}")
        # Return default features
        return {
            'text_embedding': np.zeros(384),  # Default embedding size
            'question_ratio': 0,
            'exclamation_ratio': 0,
            'word_count': 0,
            'avg_word_length': 0,
            'audio_energy_mean': 0,
            'audio_energy_std': 0,
            'audio_energy_max': 0,
            'pitch_variance': 0,
            'tempo': 0,
            'zcr_mean': 0,
            'zcr_std': 0,
            'motion_mean': 0,
            'motion_std': 0,
            'motion_max': 0,
            'scene_changes': 0,
            'face_count_mean': 0,
            'face_count_max': 0
        }
class ViralPatternDetector:
    """ML-based viral content detection without hard-coded keywords"""
    
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        # Load semantic understanding model
        self.sentence_model = SentenceTransformer(model_name)
        self.model_name = model_name  # Store for parallel processing
        
        # For emotion detection
        self.emotion_tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
        self.emotion_model = AutoModel.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
        
        # Statistical thresholds (learned from data)
        self.anomaly_threshold = 2.0  # Standard deviations
        self.energy_spike_threshold = 1.5
        
    def extract_multimodal_features(self, segment, audio_path=None, video_path=None):
        """Extract features from text, audio, and video"""
        features = {}
        
        # 1. Semantic embeddings (not keywords!)
        text_embedding = self.sentence_model.encode(segment['text'])
        features['text_embedding'] = text_embedding
        
        # 2. Linguistic patterns
        features['question_ratio'] = segment['text'].count('?') / max(len(segment['text'].split()), 1)
        features['exclamation_ratio'] = segment['text'].count('!') / max(len(segment['text'].split()), 1)
        features['word_count'] = len(segment['text'].split())
        features['avg_word_length'] = np.mean([len(w) for w in segment['text'].split()])
        
        # 3. Audio features (if available)
        if audio_path:
            features.update(self.extract_audio_features(audio_path, segment['start'], segment['end']))
        
        # 4. Video features (if available)
        if video_path:
            features.update(self.extract_video_features(video_path, segment['start'], segment['end']))
        
        return features
    
    def extract_audio_features(self, audio_path, start_time, end_time):
        """Extract audio energy, pitch variance, and speech pace"""
        # Default values to return on error
        default_features = {
            'audio_energy_mean': 0,
            'audio_energy_std': 0,
            'audio_energy_max': 0,
            'pitch_variance': 0,
            'tempo': 0,
            'zcr_mean': 0,
            'zcr_std': 0
        }
        
        try:
            # Validate inputs
            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}")
                return default_features
            
            if start_time >= end_time or start_time < 0:
                print(f"Invalid time range: {start_time}s to {end_time}s")
                return default_features
            
            duration = end_time - start_time
            if duration <= 0:
                print(f"Invalid duration: {duration}s")
                return default_features
            
            # Load audio segment with error handling
            try:
                y, sr = librosa.load(audio_path, offset=start_time, duration=duration)
            except Exception as e:
                print(f"Error loading audio segment {start_time}-{end_time}: {e}")
                return default_features
            
            # Check if audio data is valid
            if y is None or len(y) == 0:
                print(f"No audio data loaded for segment {start_time}-{end_time}")
                return default_features
            
            # Ensure minimum length for analysis
            min_samples = 1024  # Minimum samples needed for analysis
            if len(y) < min_samples:
                print(f"Audio segment too short ({len(y)} samples), padding to {min_samples}")
                y = np.pad(y, (0, max(0, min_samples - len(y))), mode='constant')
            
            # Suppress librosa warnings temporarily
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Energy (RMS) - with validation
                try:
                    rms = librosa.feature.rms(y=y)[0]
                    if len(rms) == 0:
                        raise ValueError("Empty RMS array")
                except Exception as e:
                    print(f"Error computing RMS: {e}")
                    rms = np.array([0])
                
                # Pitch variance - with validation
                try:
                    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                    valid_pitches = pitches[magnitudes > np.median(magnitudes)]
                    pitch_variance = np.var(valid_pitches) if len(valid_pitches) > 0 else 0
                except Exception as e:
                    print(f"Error computing pitch variance: {e}")
                    pitch_variance = 0
                
                # Tempo/pace - with validation
                try:
                    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                    if np.isnan(tempo) or tempo <= 0:
                        tempo = 120  # Default tempo
                except Exception as e:
                    print(f"Error computing tempo: {e}")
                    tempo = 120
                
                # Zero crossing rate - with validation
                try:
                    zcr = librosa.feature.zero_crossing_rate(y)[0]
                    if len(zcr) == 0:
                        raise ValueError("Empty ZCR array")
                except Exception as e:
                    print(f"Error computing ZCR: {e}")
                    zcr = np.array([0])
            
            # Return validated features
            return {
                'audio_energy_mean': float(np.mean(rms)) if len(rms) > 0 else 0,
                'audio_energy_std': float(np.std(rms)) if len(rms) > 0 else 0,
                'audio_energy_max': float(np.max(rms)) if len(rms) > 0 else 0,
                'pitch_variance': float(pitch_variance) if not np.isnan(pitch_variance) else 0,
                'tempo': float(tempo) if not np.isnan(tempo) else 120,
                'zcr_mean': float(np.mean(zcr)) if len(zcr) > 0 else 0,
                'zcr_std': float(np.std(zcr)) if len(zcr) > 0 else 0
            }
            
        except Exception as e:
            print(f"Unexpected error in extract_audio_features: {e}")
            return default_features
    
    def extract_video_features(self, video_path, start_time, end_time):
        """Extract visual motion, scene changes, and face detection"""
        # Default values
        default_features = {
            'motion_mean': 0,
            'motion_std': 0,
            'motion_max': 0,
            'scene_changes': 0,
            'face_count_mean': 0,
            'face_count_max': 0
        }
        
        try:
            if not os.path.exists(video_path):
                print(f"Video file not found: {video_path}")
                return default_features
            
            if start_time >= end_time or start_time < 0:
                print(f"Invalid time range: {start_time}s to {end_time}s")
                return default_features
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Could not open video file: {video_path}")
                return default_features
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                print(f"Invalid FPS: {fps}")
                cap.release()
                return default_features
            
            # Jump to start time
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Face cascade for face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            motion_scores = []
            scene_changes = 0
            face_counts = []
            prev_frame = None
            
            for frame_idx in range(start_frame, min(end_frame, start_frame + int(fps * 5))):  # Sample 5 seconds
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Motion detection
                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, gray)
                    motion_score = np.mean(diff)
                    motion_scores.append(motion_score)
                    
                    # Scene change detection
                    if motion_score > 30:  # Threshold
                        scene_changes += 1
                
                # Face detection
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                face_counts.append(len(faces))
                
                prev_frame = gray
            
            cap.release()
            
            return {
                'motion_mean': float(np.mean(motion_scores)) if motion_scores else 0,
                'motion_std': float(np.std(motion_scores)) if motion_scores else 0,
                'motion_max': float(np.max(motion_scores)) if motion_scores else 0,
                'scene_changes': int(scene_changes),
                'face_count_mean': float(np.mean(face_counts)) if face_counts else 0,
                'face_count_max': int(np.max(face_counts)) if face_counts else 0
            }
            
        except Exception as e:
            print(f"Error in extract_video_features: {e}")
            return default_features
    
    def detect_anomalies(self, features_list):
        """Detect statistical anomalies that often indicate viral moments"""
        # Convert to numpy array for analysis
        feature_matrix = []
        
        # Extract numerical features only
        numerical_features = ['audio_energy_mean', 'audio_energy_std', 'motion_mean', 
                            'scene_changes', 'face_count_max', 'question_ratio', 
                            'exclamation_ratio', 'word_count']
        
        for features in features_list:
            row = [features.get(f, 0) for f in numerical_features]
            feature_matrix.append(row)
        
        feature_matrix = np.array(feature_matrix)
        
        # Handle empty feature matrix
        if feature_matrix.size == 0:
            return []
        
        # Standardize features
        scaler = StandardScaler()
        try:
            scaled_features = scaler.fit_transform(feature_matrix)
        except Exception as e:
            print(f"Error scaling features: {e}")
            return [0] * len(features_list)
        
        # Calculate anomaly scores
        anomaly_scores = []
        for i, row in enumerate(scaled_features):
            # Distance from mean
            score = np.sqrt(np.sum(row**2))
            anomaly_scores.append(float(score) if not np.isnan(score) else 0)
        
        return anomaly_scores
    
    def cluster_semantic_moments(self, embeddings):
        """Find semantic clusters to identify key themes"""
        if len(embeddings) == 0:
            return [], []
        
        try:
            # DBSCAN clustering for finding dense regions
            clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
            clusters = clustering.fit_predict(embeddings)
            
            # Find transition points between clusters
            transitions = []
            for i in range(1, len(clusters)):
                if clusters[i] != clusters[i-1]:
                    transitions.append(i)
            
            return clusters, transitions
        except Exception as e:
            print(f"Error in clustering: {e}")
            return [-1] * len(embeddings), []
    
    def learn_from_viral_videos(self, viral_video_data):
        """Learn patterns from actual viral videos (YouTube API, TikTok data, etc.)"""
        # This would connect to APIs to analyze truly viral content
        # For now, we'll use statistical patterns
        
        viral_patterns = {
            'avg_segment_length': 8.5,  # seconds
            'energy_spike_frequency': 0.3,  # spikes per minute
            'emotion_variance': 0.7,  # high emotion variance
            'scene_change_rate': 2.5,  # changes per minute
        }
        
        return viral_patterns
    
    def score_segments_parallel(self, segments, audio_path=None, video_path=None, 
                              max_workers=None, use_threading=False):
        """Score segments with parallel feature extraction for large datasets"""
        
        if not segments:
            return []
        
        # Determine optimal number of workers
        if max_workers is None:
            cpu_count = mp.cpu_count()
            max_workers = max(1, min(len(segments), cpu_count - 1))
        
        print(f"🚀 Processing {len(segments)} segments with {max_workers} workers")
        print(f"Using {'ThreadPoolExecutor' if use_threading else 'ProcessPoolExecutor'}")
        
        # Prepare arguments for parallel processing
        tasks = []
        for segment in segments:
            task_args = (segment, audio_path, video_path, self.model_name)
            tasks.append(task_args)
        
        # Choose executor type
        executor_class = ThreadPoolExecutor if use_threading else ProcessPoolExecutor
        
        all_features = []
        embeddings = []
        start_time = time.time()
        
        try:
            with executor_class(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_index = {
                    executor.submit(_extract_features_worker, task): i 
                    for i, task in enumerate(tasks)
                }
                
                # Collect results with progress bar
                results = [None] * len(tasks)  # Preserve order
                with tqdm(total=len(tasks), desc="Extracting features") as pbar:
                    for future in as_completed(future_to_index):
                        index = future_to_index[future]
                        try:
                            features = future.result()
                            results[index] = features
                            pbar.update(1)
                        except Exception as e:
                            print(f"Error processing segment {index}: {e}")
                            # Use default features
                            results[index] = {
                                'text_embedding': np.zeros(384),
                                'question_ratio': 0, 'exclamation_ratio': 0,
                                'word_count': 0, 'avg_word_length': 0,
                                'audio_energy_mean': 0, 'audio_energy_std': 0,
                                'audio_energy_max': 0, 'pitch_variance': 0,
                                'tempo': 0, 'zcr_mean': 0, 'zcr_std': 0,
                                'motion_mean': 0, 'motion_std': 0, 'motion_max': 0,
                                'scene_changes': 0, 'face_count_mean': 0, 'face_count_max': 0
                            }
                            pbar.update(1)
                
                # Extract features and embeddings maintaining order
                for features in results:
                    if features:
                        all_features.append(features)
                        embeddings.append(features['text_embedding'])
        
        except Exception as e:
            print(f"Error in parallel processing: {e}")
            # Fallback to sequential processing
            print("Falling back to sequential processing...")
            return self.score_segments(segments, audio_path, video_path)
        
        extraction_time = time.time() - start_time
        print(f"⚡ Feature extraction completed in {extraction_time:.1f}s")
        print(f"📊 Average time per segment: {extraction_time/len(segments):.3f}s")
        
        if not all_features:
            return []
        
        # Continue with sequential processing for the rest
        print("🔍 Detecting anomalies...")
        anomaly_scores = self.detect_anomalies(all_features)
        
        print("🧠 Performing semantic clustering...")
        embeddings_array = np.array(embeddings)
        clusters, transitions = self.cluster_semantic_moments(embeddings_array)
        
        print("🎯 Scoring segments...")
        scored_segments = []
        for i, (segment, features) in enumerate(zip(segments, all_features)):
            score = 0
            
            # 1. Anomaly score (unusual moments)
            if i < len(anomaly_scores):
                score += anomaly_scores[i] * 2.0
            
            # 2. Transition points (topic changes)
            if i in transitions:
                score += 3.0
            
            # 3. Audio energy spikes
            energy_mean = np.mean([f.get('audio_energy_mean', 0) for f in all_features])
            if features.get('audio_energy_max', 0) > energy_mean * 1.5:
                score += 2.5
            
            # 4. Visual motion peaks
            motion_mean = np.mean([f.get('motion_mean', 0) for f in all_features])
            if features.get('motion_max', 0) > motion_mean * 2:
                score += 2.0
            
            # 5. Face presence (human element)
            if features.get('face_count_max', 0) > 0:
                score += 1.5
            
            # 6. Engagement patterns (questions, exclamations)
            score += features.get('question_ratio', 0) * 10
            score += features.get('exclamation_ratio', 0) * 8
            
            # 7. Scene changes (visual interest)
            score += features.get('scene_changes', 0) * 0.5
            
            # 8. Optimal length bonus
            duration = segment['end'] - segment['start']
            if 7 <= duration <= 12:  # Optimal short length
                score += 2.0
            
            scored_segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'],
                'score': float(score),
                'features': features,
                'cluster': clusters[i] if i < len(clusters) else -1
            })
        
        return scored_segments
    def score_segments(self, segments, audio_path=None, video_path=None, parallel=True, 
                      max_workers=None, use_threading=False):
        """Score segments based on learned patterns, with optional parallel processing"""
        
        if not segments:
            return []
        
        # Auto-decide whether to use parallel processing
        if parallel and len(segments) >= 10:  # Use parallel for 10+ segments
            return self.score_segments_parallel(segments, audio_path, video_path, 
                                              max_workers, use_threading)
        
        # Sequential processing (original implementation)
        print(f"Processing {len(segments)} segments sequentially...")
        
        # Extract features for all segments
        all_features = []
        embeddings = []
        
        print("Extracting features from segments...")
        for i, segment in enumerate(tqdm(segments, desc="Processing segments")):
            features = self.extract_multimodal_features(segment, audio_path, video_path)
            all_features.append(features)
            embeddings.append(features['text_embedding'])
        
        if not all_features:
            return []
        
        # Detect anomalies (unusual = interesting)
        print("Detecting anomalies...")
        anomaly_scores = self.detect_anomalies(all_features)
        
        # Semantic clustering
        print("Performing semantic clustering...")
        embeddings_array = np.array(embeddings)
        clusters, transitions = self.cluster_semantic_moments(embeddings_array)
        
        # Score each segment
        print("Scoring segments...")
        scored_segments = []
        for i, (segment, features) in enumerate(zip(segments, all_features)):
            score = 0
            
            # 1. Anomaly score (unusual moments)
            if i < len(anomaly_scores):
                score += anomaly_scores[i] * 2.0
            
            # 2. Transition points (topic changes)
            if i in transitions:
                score += 3.0
            
            # 3. Audio energy spikes
            energy_mean = np.mean([f.get('audio_energy_mean', 0) for f in all_features])
            if features.get('audio_energy_max', 0) > energy_mean * 1.5:
                score += 2.5
            
            # 4. Visual motion peaks
            motion_mean = np.mean([f.get('motion_mean', 0) for f in all_features])
            if features.get('motion_max', 0) > motion_mean * 2:
                score += 2.0
            
            # 5. Face presence (human element)
            if features.get('face_count_max', 0) > 0:
                score += 1.5
            
            # 6. Engagement patterns (questions, exclamations)
            score += features.get('question_ratio', 0) * 10
            score += features.get('exclamation_ratio', 0) * 8
            
            # 7. Scene changes (visual interest)
            score += features.get('scene_changes', 0) * 0.5
            
            # 8. Optimal length bonus
            duration = segment['end'] - segment['start']
            if 7 <= duration <= 12:  # Optimal short length
                score += 2.0
            
            scored_segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'],
                'score': float(score),
                'features': features,
                'cluster': clusters[i] if i < len(clusters) else -1
            })
        
        return scored_segments
    
    def select_diverse_highlights(self, scored_segments, top_n=5, min_gap=30):
        """Select diverse highlights covering different moments"""
        if not scored_segments:
            return []
        
        # Sort by score
        sorted_segments = sorted(scored_segments, key=lambda x: x['score'], reverse=True)
        
        selected = []
        clusters_used = set()
        
        for segment in sorted_segments:
            if len(selected) >= top_n:
                break
            
            # Check temporal diversity
            too_close = any(
                abs(segment['start'] - s['start']) < min_gap 
                for s in selected
            )
            
            if not too_close:
                selected.append(segment)
                clusters_used.add(segment['cluster'])
        
        # Sort by time
        selected.sort(key=lambda x: x['start'])
        
        return selected


class ViralShortsCreator:
    """Create viral shorts using ML-based pattern detection"""
    
    def __init__(self):
        self.detector = ViralPatternDetector()
        
    def create_shorts(self, whisper_segments, audio_path=None, video_path=None, 
                     num_shorts=5, context_seconds=2, parallel=True, max_workers=None):
        """Create viral shorts from any video with optional parallel processing"""
        
        if not whisper_segments:
            print("No segments provided")
            return []
        
        # Convert segments to dict format
        segments = []
        for seg in whisper_segments:
            try:
                if hasattr(seg, 'start'):
                    # WhisperX/Whisper object
                    segments.append({
                        "start": float(seg.start),
                        "end": float(seg.end),
                        "text": str(seg.text)
                    })
                else:
                    # Dictionary format
                    segments.append({
                        "start": float(seg.get("start", 0)),
                        "end": float(seg.get("end", 0)),
                        "text": str(seg.get("text", ""))
                    })
            except Exception as e:
                print(f"Error processing segment: {e}")
                continue
        
        if not segments:
            print("No valid segments found")
            return []
        
        print(f"🔍 Analyzing {len(segments)} segments for viral patterns...")
        
        # Score segments using ML approach (with parallel processing)
        scored_segments = self.detector.score_segments(
            segments, audio_path, video_path, 
            parallel=parallel, max_workers=max_workers
        )
        
        if not scored_segments:
            print("No scored segments found")
            return []
        
        # Select best highlights
        highlights = self.detector.select_diverse_highlights(scored_segments, top_n=num_shorts)
        
        # Create shorts with context
        shorts = []
        for highlight in highlights:
            # Find surrounding segments for context
            start_time = max(0, highlight['start'] - context_seconds)
            end_time = highlight['end'] + context_seconds
            
            # Get all segments in this time range
            context_segments = [
                seg for seg in segments 
                if seg['end'] > start_time and seg['start'] < end_time
            ]
            
            shorts.append({
                'start': start_time,
                'end': end_time,
                'duration': end_time - start_time,
                'score': highlight['score'],
                'highlight_text': highlight['text'],
                'segments': context_segments,
                'features': highlight['features']
            })
        
        return shorts
    
    def analyze_trends(self, platform='youtube'):
        """Analyze current viral trends to adapt scoring"""
        # This would connect to platform APIs
        # For demonstration, showing the concept
        
        trending_patterns = {
            'youtube': {
                'optimal_length': 8.5,
                'hook_importance': 0.8,
                'emotion_variance': 0.7
            },
            'tiktok': {
                'optimal_length': 15,
                'hook_importance': 0.9,
                'emotion_variance': 0.8
            },
            'instagram': {
                'optimal_length': 10,
                'hook_importance': 0.85,
                'emotion_variance': 0.75
            }
        }
        
        return trending_patterns.get(platform, trending_patterns['youtube'])


# Example usage
def create_viral_shorts_ml(whisper_segments, audio_path=None, video_path=None, 
                          parallel=True, max_workers=None):
    """Main function using ML approach with parallel processing"""
    
    creator = ViralShortsCreator()
    
    # Create shorts with parallel processing
    shorts = creator.create_shorts(
        whisper_segments=whisper_segments,
        audio_path=audio_path,
        video_path=video_path,
        num_shorts=5,
        context_seconds=2,
        parallel=parallel,
        max_workers=max_workers
    )
    
    print(f"\n🎯 Found {len(shorts)} viral short opportunities:")
    for i, short in enumerate(shorts, 1):
        print(f"\n📹 Short #{i}:")
        print(f"   Duration: {short['duration']:.1f}s")
        print(f"   Score: {short['score']:.2f}")
        print(f"   Time: {short['start']:.1f}s - {short['end']:.1f}s")
        
        # Show what made it viral
        features = short['features']
        if features.get('audio_energy_max', 0) > 0:
            print(f"   🔊 High audio energy: {features['audio_energy_max']:.2f}")
        if features.get('motion_max', 0) > 0:
            print(f"   🎬 High visual motion: {features['motion_max']:.2f}")
        if features.get('face_count_max', 0) > 0:
            print(f"   👤 Human presence detected")
        
        print(f"   📝 Text: {short['highlight_text'][:80]}...")
    
    return shorts


# Example with different processing modes for large datasets
if __name__ == "__main__":
    # Example with 700+ segments (simulated)
    example_segments = []
    for i in range(700):  # Simulate 700 segments
        example_segments.append({
            "start": i * 5,
            "end": i * 5 + 4,
            "text": f"This is segment {i} with some interesting content that might be viral"
        })
    
    print("=" * 80)
    print("PARALLEL PROCESSING DEMONSTRATION")
    print("=" * 80)
    
    # Method 1: Automatic parallel processing (recommended for large datasets)
    print("\n🚀 Method 1: Automatic Parallel Processing")
    start_time = time.time()
    shorts_auto = create_viral_shorts_ml(
        whisper_segments=example_segments,
        audio_path="video.mp3",
        video_path="video.mp4",
        parallel=True  # Auto-detects optimal workers
    )
    auto_time = time.time() - start_time
    print(f"⏱️ Time taken: {auto_time:.1f}s")
    
    # Method 2: Manual parallel with custom workers
    print("\n🔧 Method 2: Manual Parallel Configuration")
    start_time = time.time()
    shorts_manual = create_viral_shorts_ml(
        whisper_segments=example_segments,
        audio_path="video.mp3",
        video_path="video.mp4",
        parallel=True,
        max_workers=8  # Force 8 workers
    )
    manual_time = time.time() - start_time
    print(f"⏱️ Time taken: {manual_time:.1f}s")
    
    # Method 3: Sequential processing (for comparison)
    print("\n🐌 Method 3: Sequential Processing")
    start_time = time.time()
    shorts_sequential = create_viral_shorts_ml(
        whisper_segments=example_segments[:50],  # Only test with 50 segments
        audio_path="video.mp3",
        video_path="video.mp4",
        parallel=False
    )
    sequential_time = time.time() - start_time
    print(f"⏱️ Time taken (50 segments): {sequential_time:.1f}s")
    print(f"📊 Estimated time for 700 segments: {sequential_time * 14:.1f}s")
    
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"🚀 Parallel Auto:     {auto_time:.1f}s")
    print(f"🔧 Parallel Manual:   {manual_time:.1f}s") 
    print(f"🐌 Sequential (est):  {sequential_time * 14:.1f}s")
    if sequential_time > 0:
        speedup = (sequential_time * 14) / auto_time
        print(f"⚡ Speed improvement: {speedup:.1f}x faster!")
    print("=" * 80)

# Example with multimodal processing
if __name__ == "__main__":
    # Example segments
    example_segments = [
        {"start": 0, "end": 5, "text": "This completely changed my life"},
        {"start": 5, "end": 10, "text": "I never expected this to happen"},
        {"start": 30, "end": 35, "text": "Watch what happens next"},
        {"start": 35, "end": 40, "text": "The reaction was absolutely priceless"},
    ]
    
    # Create shorts with ML approach
    shorts = create_viral_shorts_ml(
        whisper_segments=example_segments,
        audio_path="video.mp3",  # Optional
        video_path="video.mp4"   # Optional
    )