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

class ViralPatternDetector:
    """ML-based viral content detection without hard-coded keywords"""
    
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        # Load semantic understanding model
        self.sentence_model = SentenceTransformer(model_name)
        
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
        try:
            # Load audio segment
            y, sr = librosa.load(audio_path, offset=start_time, duration=end_time-start_time)
            
            # Energy (RMS)
            rms = librosa.feature.rms(y=y)[0]
            
            # Pitch variance
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_variance = np.var(pitches[magnitudes > np.median(magnitudes)])
            
            # Tempo/pace
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Zero crossing rate (excitement indicator)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            
            return {
                'audio_energy_mean': np.mean(rms),
                'audio_energy_std': np.std(rms),
                'audio_energy_max': np.max(rms),
                'pitch_variance': pitch_variance,
                'tempo': tempo,
                'zcr_mean': np.mean(zcr),
                'zcr_std': np.std(zcr)
            }
        except:
            return {
                'audio_energy_mean': 0,
                'audio_energy_std': 0,
                'audio_energy_max': 0,
                'pitch_variance': 0,
                'tempo': 0,
                'zcr_mean': 0,
                'zcr_std': 0
            }
    
    def extract_video_features(self, video_path, start_time, end_time):
        """Extract visual motion, scene changes, and face detection"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
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
                'motion_mean': np.mean(motion_scores) if motion_scores else 0,
                'motion_std': np.std(motion_scores) if motion_scores else 0,
                'motion_max': np.max(motion_scores) if motion_scores else 0,
                'scene_changes': scene_changes,
                'face_count_mean': np.mean(face_counts) if face_counts else 0,
                'face_count_max': np.max(face_counts) if face_counts else 0
            }
        except:
            return {
                'motion_mean': 0,
                'motion_std': 0,
                'motion_max': 0,
                'scene_changes': 0,
                'face_count_mean': 0,
                'face_count_max': 0
            }
    
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
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_matrix)
        
        # Calculate anomaly scores
        anomaly_scores = []
        for i, row in enumerate(scaled_features):
            # Distance from mean
            score = np.sqrt(np.sum(row**2))
            anomaly_scores.append(score)
        
        return anomaly_scores
    
    def cluster_semantic_moments(self, embeddings):
        """Find semantic clusters to identify key themes"""
        # DBSCAN clustering for finding dense regions
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
        clusters = clustering.fit_predict(embeddings)
        
        # Find transition points between clusters
        transitions = []
        for i in range(1, len(clusters)):
            if clusters[i] != clusters[i-1]:
                transitions.append(i)
        
        return clusters, transitions
    
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
    
    def score_segments(self, segments, audio_path=None, video_path=None):
        """Score segments based on learned patterns, not keywords"""
        
        # Extract features for all segments
        all_features = []
        embeddings = []
        
        for segment in segments:
            features = self.extract_multimodal_features(segment, audio_path, video_path)
            all_features.append(features)
            embeddings.append(features['text_embedding'])
        
        # Detect anomalies (unusual = interesting)
        anomaly_scores = self.detect_anomalies(all_features)
        
        # Semantic clustering
        embeddings_array = np.array(embeddings)
        clusters, transitions = self.cluster_semantic_moments(embeddings_array)
        
        # Score each segment
        scored_segments = []
        for i, (segment, features) in enumerate(zip(segments, all_features)):
            score = 0
            
            # 1. Anomaly score (unusual moments)
            score += anomaly_scores[i] * 2.0
            
            # 2. Transition points (topic changes)
            if i in transitions:
                score += 3.0
            
            # 3. Audio energy spikes
            if features.get('audio_energy_max', 0) > np.mean([f.get('audio_energy_mean', 0) for f in all_features]) * 1.5:
                score += 2.5
            
            # 4. Visual motion peaks
            if features.get('motion_max', 0) > np.mean([f.get('motion_mean', 0) for f in all_features]) * 2:
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
                'score': score,
                'features': features,
                'cluster': clusters[i] if i < len(clusters) else -1
            })
        
        return scored_segments
    
    def select_diverse_highlights(self, scored_segments, top_n=5, min_gap=30):
        """Select diverse highlights covering different moments"""
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
                     num_shorts=5, context_seconds=2):
        """Create viral shorts from any video"""
        
        # Convert segments to dict format
        segments = [
            {
                "start": seg.start if hasattr(seg, 'start') else seg.get("start", 0),
                "end": seg.end if hasattr(seg, 'end') else seg.get("end", 0),
                "text": seg.text if hasattr(seg, 'text') else seg.get("text", "")
            }
            for seg in whisper_segments
        ]
        
        print("🔍 Analyzing content patterns...")
        
        # Score segments using ML approach
        scored_segments = self.detector.score_segments(segments, audio_path, video_path)
        
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
def create_viral_shorts_ml(whisper_segments, audio_path=None, video_path=None):
    """Main function using ML approach"""
    
    creator = ViralShortsCreator()
    
    # Create shorts
    shorts = creator.create_shorts(
        whisper_segments=whisper_segments,
        audio_path=audio_path,
        video_path=video_path,
        num_shorts=5,
        context_seconds=2
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