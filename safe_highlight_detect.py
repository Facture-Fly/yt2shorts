import torch
import os
# Disable meta device for transformers to avoid meta tensor issues
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
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
import psutil
import gc
import threading
from queue import Queue
import numpy as np
import shutil

def score_segments(self, segments, audio_path=None, video_path=None, 
                      processing_mode='auto', max_workers=None, batch_size=32):
        """
        Score segments with multiple processing modes
        
        processing_mode options:
        - 'auto': Automatically choose best method based on system resources
        - 'gpu': Use GPU acceleration with batching (recommended for 100+ segments)
        - 'memory_safe': Conservative CPU processing (safe for any system)
        - 'sequential': Original sequential processing
        """
        
        if not segments:
            return []
        
        segment_count = len(segments)
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        has_gpu = torch.cuda.is_available()
        
        print(f"📊 Processing {segment_count} segments")
        print(f"💾 Available memory: {available_memory_gb:.1f} GB")
        print(f"🎮 GPU available: {has_gpu}")
        
        # Auto-select processing mode
        if processing_mode == 'auto':
            if has_gpu and segment_count >= 100 and available_memory_gb >= 6:
                processing_mode = 'gpu'
            elif segment_count >= 50 and available_memory_gb >= 4:
                processing_mode = 'memory_safe'
            else:
                processing_mode = 'sequential'
        
        print(f"🚀 Using {processing_mode} processing mode")
        
        # Route to appropriate processing method
        if processing_mode == 'gpu' and has_gpu:
            return self.score_segments_gpu_batched(segments, audio_path, video_path, batch_size, max_workers or 2)
        elif processing_mode == 'memory_safe':
            import numpy as np
            
def _extract_features_worker_gpu(args):
    """
    GPU-accelerated worker function for parallel feature extraction
    """
    segment, audio_path, video_path, model_name, device, batch_idx, total_batches = args
    
    try:
        # Check memory before processing
        if psutil.virtual_memory().percent > 85:
            print(f"⚠️ High memory usage ({psutil.virtual_memory().percent:.1f}%), skipping segment")
            return None
        
        # Create detector instance - force CPU in workers to avoid meta tensor issues
        detector = ViralPatternDetector(model_name=model_name, device='cpu')
        features = detector.extract_multimodal_features(segment, audio_path, video_path)
        
        # Clear GPU cache periodically
        if hasattr(torch.cuda, 'empty_cache') and batch_idx % 10 == 0:
            torch.cuda.empty_cache()
        
        return features
        
    except Exception as e:
        print(f"Error in GPU worker processing segment {segment.get('start', 0):.1f}s: {e}")
        return None
    finally:
        # Force garbage collection
        gc.collect()


def _extract_features_worker(args):
    """
    CPU worker function for parallel feature extraction (memory-safe version)
    """
    segment, audio_path, video_path, model_name = args
    
    try:
        # Check system resources before processing
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 80:
            print(f"⚠️ High memory usage ({memory_percent:.1f}%), waiting...")
            time.sleep(1)
            if psutil.virtual_memory().percent > 85:
                print("❌ Memory too high, skipping segment")
                return None
        
        # Create detector instance in worker - force CPU to avoid meta tensor issues
        detector = ViralPatternDetector(model_name=model_name, device='cpu')
        features = detector.extract_multimodal_features(segment, audio_path, video_path)
        return features
        
    except Exception as e:
        print(f"Error processing segment {segment.get('start', 0):.1f}s: {e}")
        return None
    finally:
        # Force cleanup
        gc.collect()
class ViralPatternDetector:
    """ML-based viral content detection with GPU support and memory management"""
    
    def __init__(self, model_name='./models/sentence-transformers/all-MiniLM-L6-v2', device=None):
        if device is None:
            device = self._detect_and_test_device(model_name)
        
        self.device = device
        self.model_name = model_name
        
        print(f"🖥️ Final device selection: {self.device}")
        
        # Load models with tested device
        self.sentence_model = SentenceTransformer(model_name, device=self.device)
        
        self.emotion_tokenizer = AutoTokenizer.from_pretrained(
            "./models/j-hartmann/emotion-english-distilroberta-base"
        )
        # Load model avoiding meta tensors completely
        try:
            self.emotion_model = AutoModel.from_pretrained(
                "./models/j-hartmann/emotion-english-distilroberta-base",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False,  # Disable to avoid meta tensors
                device_map=None  # Disable device_map to avoid meta tensors
            )
            # Move to device after loading
            self.emotion_model = self.emotion_model.to(self.device)
        except Exception as e:
            print(f"⚠️ Could not load emotion model on {self.device}: {e}")
            print("🔄 Falling back to CPU-only loading...")
            try:
                self.emotion_model = AutoModel.from_pretrained(
                    "./models/j-hartmann/emotion-english-distilroberta-base",
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=False
                )
                self.emotion_model = self.emotion_model.to('cpu')
                self.device = 'cpu'  # Force CPU if GPU loading fails
            except Exception as e2:
                print(f"❌ Final fallback failed: {e2}")
                raise e2
        
        self.anomaly_threshold = 2.0
        self.energy_spike_threshold = 1.5
    
    def _detect_and_test_device(self, model_name):
        """Detect GPU and test if it actually works with our models"""
        
        if not torch.cuda.is_available():
            print("🖥️ No GPU detected - using CPU")
            return 'cpu'
        
        # Get GPU info
        device_name = torch.cuda.get_device_name(0)
        print(f"🎮 GPU detected: {device_name}")
        
        # Check if it's AMD
        is_amd = 'AMD' in device_name or 'Radeon' in device_name
        
        if is_amd:
            print("🔴 AMD GPU detected - testing compatibility...")
            
            try:
                # Test basic operations first
                test_tensor = torch.randn(100, 100).cuda()
                result = torch.mm(test_tensor, test_tensor)
                print("✅ Basic GPU operations work")
                
                # Test SentenceTransformer specifically
                print("🧪 Testing SentenceTransformer on AMD GPU...")
                test_model = SentenceTransformer(model_name, device='cuda')
                test_embedding = test_model.encode("test", show_progress_bar=False)
                print("✅ SentenceTransformer works on AMD GPU!")
                
                return 'cuda'
                
            except Exception as e:
                print(f"❌ AMD GPU test failed: {e}")
                print("🔄 AMD GPU detected but incompatible - falling back to CPU")
                print("   This is common with ROCm and certain model operations")
                return 'cpu'
        else:
            print("🟢 NVIDIA GPU detected - should work fine")
            return 'cuda'
        
    def extract_multimodal_features(self, segment, audio_path=None, video_path=None):
        """Extract features from text, audio, and video with GPU acceleration"""
        features = {}
        
        try:
            # 1. Semantic embeddings (GPU accelerated!)
            with torch.no_grad():  # Save GPU memory
                text_embedding = self.sentence_model.encode(
                    segment['text'], 
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
            features['text_embedding'] = text_embedding
            
            # 2. Linguistic patterns (CPU-based, fast)
            text = segment['text']
            word_count = len(text.split())
            features['question_ratio'] = text.count('?') / max(word_count, 1)
            features['exclamation_ratio'] = text.count('!') / max(word_count, 1)
            features['word_count'] = word_count
            features['avg_word_length'] = np.mean([len(w) for w in text.split()]) if word_count > 0 else 0
            
            # 3. Audio features (if available) - keep on CPU
            if audio_path:
                features.update(self.extract_audio_features(audio_path, segment['start'], segment['end']))
            else:
                features.update({
                    'audio_energy_mean': 0, 'audio_energy_std': 0, 'audio_energy_max': 0,
                    'pitch_variance': 0, 'tempo': 0, 'zcr_mean': 0, 'zcr_std': 0
                })
            
            # 4. Video features (if available) - keep on CPU  
            if video_path:
                features.update(self.extract_video_features(video_path, segment['start'], segment['end']))
            else:
                features.update({
                    'motion_mean': 0, 'motion_std': 0, 'motion_max': 0,
                    'scene_changes': 0, 'face_count_mean': 0, 'face_count_max': 0
                })
            
            return features
            
        except Exception as e:
            print(f"Error in extract_multimodal_features: {e}")
            # Return default features
            return {
                'text_embedding': np.zeros(384),
                'question_ratio': 0, 'exclamation_ratio': 0, 'word_count': 0, 'avg_word_length': 0,
                'audio_energy_mean': 0, 'audio_energy_std': 0, 'audio_energy_max': 0,
                'pitch_variance': 0, 'tempo': 0, 'zcr_mean': 0, 'zcr_std': 0,
                'motion_mean': 0, 'motion_std': 0, 'motion_max': 0,
                'scene_changes': 0, 'face_count_mean': 0, 'face_count_max': 0
            }
    
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
    
    def score_segments_gpu_batched(self, segments, audio_path=None, video_path=None, 
                                   batch_size=32, max_workers=2):
        """GPU-accelerated batched processing to prevent system crashes"""
        
        if not segments:
            return []
        
        print(f"🚀 GPU Batched Processing: {len(segments)} segments")
        print(f"📦 Batch size: {batch_size}, Workers: {max_workers}")
        
        # Check GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🎮 GPU Memory: {gpu_memory:.1f} GB")
        
        all_features = []
        embeddings = []
        
        # Process in batches to avoid memory issues
        for batch_start in tqdm(range(0, len(segments), batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, len(segments))
            batch_segments = segments[batch_start:batch_end]
            
            # Check system resources before each batch
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 75:
                print(f"⚠️ High memory usage ({memory_percent:.1f}%), reducing batch size")
                batch_size = max(8, batch_size // 2)
                time.sleep(2)  # Let system recover
            
            # Process batch with limited workers
            batch_features = self._process_batch_gpu(
                batch_segments, audio_path, video_path, max_workers
            )
            
            all_features.extend(batch_features)
            embeddings.extend([f['text_embedding'] for f in batch_features if f])
            
            # Clear GPU cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Small delay to prevent system overload
            time.sleep(0.1)
        
        return self._score_from_features(segments, all_features, embeddings)
    
    def _process_batch_gpu(self, batch_segments, audio_path, video_path, max_workers):
        """Process a batch of segments with GPU acceleration"""
        
        batch_features = []
        
        # For small batches, process sequentially to avoid overhead
        if len(batch_segments) <= 4:
            for segment in batch_segments:
                features = self.extract_multimodal_features(segment, audio_path, video_path)
                batch_features.append(features)
            return batch_features
        
        # Parallel processing for larger batches
        tasks = []
        for i, segment in enumerate(batch_segments):
            task_args = (segment, audio_path, video_path, self.model_name, 
                        self.device, i, len(batch_segments))
            tasks.append(task_args)
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(_extract_features_worker_gpu, task): i 
                    for i, task in enumerate(tasks)
                }
                
                results = [None] * len(tasks)
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        features = future.result()
                        results[index] = features if features else self._get_default_features()
                    except Exception as e:
                        print(f"Batch error: {e}")
                        results[index] = self._get_default_features()
                
                batch_features = [r for r in results if r]
        
        except Exception as e:
            print(f"Batch processing failed: {e}, falling back to sequential")
            for segment in batch_segments:
                features = self.extract_multimodal_features(segment, audio_path, video_path)
                batch_features.append(features)
        
        return batch_features
    
    def score_segments_memory_safe(self, segments, audio_path=None, video_path=None, 
                                  max_workers=None, chunk_size=50):
        """Memory-safe processing for large datasets"""
        
        if not segments:
            return []
        
        # Auto-determine safe parameters based on system resources
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        cpu_count = mp.cpu_count()
        
        # Conservative settings to prevent crashes
        if available_memory_gb < 4:
            max_workers = 1
            chunk_size = 20
            print("⚠️ Low memory detected, using conservative settings")
        elif available_memory_gb < 8:
            max_workers = min(2, cpu_count - 2)
            chunk_size = 30
        else:
            max_workers = max_workers or min(4, cpu_count - 1)
            chunk_size = min(chunk_size, 50)
        
        print(f"🛡️ Memory-safe processing: {max_workers} workers, {chunk_size} chunk size")
        print(f"💾 Available memory: {available_memory_gb:.1f} GB")
        
        all_features = []
        embeddings = []
        
        # Process in chunks
        for chunk_start in tqdm(range(0, len(segments), chunk_size), desc="Processing chunks"):
            chunk_end = min(chunk_start + chunk_size, len(segments))
            chunk_segments = segments[chunk_start:chunk_end]
            
            # Monitor memory before processing chunk
            memory_before = psutil.virtual_memory().percent
            if memory_before > 80:
                print(f"⚠️ High memory usage ({memory_before:.1f}%), waiting...")
                time.sleep(3)
                gc.collect()
            
            # Process chunk
            chunk_features = self._process_chunk_safe(
                chunk_segments, audio_path, video_path, max_workers
            )
            
            all_features.extend(chunk_features)
            embeddings.extend([f['text_embedding'] for f in chunk_features if f])
            
            # Clean up after chunk
            gc.collect()
            time.sleep(0.2)
        
        return self._score_from_features(segments, all_features, embeddings)
    
    def _process_chunk_safe(self, chunk_segments, audio_path, video_path, max_workers):
        """Safely process a chunk of segments"""
        chunk_features = []
        
        if max_workers == 1 or len(chunk_segments) <= 3:
            # Sequential processing
            for segment in chunk_segments:
                features = self.extract_multimodal_features(segment, audio_path, video_path)
                chunk_features.append(features)
        else:
            # Limited parallel processing
            tasks = [(seg, audio_path, video_path, self.model_name) for seg in chunk_segments]
            
            try:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(_extract_features_worker, tasks))
                    chunk_features = [r if r else self._get_default_features() for r in results]
            except Exception as e:
                print(f"Chunk processing failed: {e}, using sequential")
                for segment in chunk_segments:
                    features = self.extract_multimodal_features(segment, audio_path, video_path)
                    chunk_features.append(features)
        
        return chunk_features
    
    def _get_default_features(self):
        """Get default feature values"""
        return {
            'text_embedding': np.zeros(384),
            'question_ratio': 0, 'exclamation_ratio': 0, 'word_count': 0, 'avg_word_length': 0,
            'audio_energy_mean': 0, 'audio_energy_std': 0, 'audio_energy_max': 0,
            'pitch_variance': 0, 'tempo': 0, 'zcr_mean': 0, 'zcr_std': 0,
            'motion_mean': 0, 'motion_std': 0, 'motion_max': 0,
            'scene_changes': 0, 'face_count_mean': 0, 'face_count_max': 0
        }
    def _score_from_features(self, segments, all_features, embeddings):
        """Score segments from extracted features"""
        if not all_features or not embeddings:
            return []
        
        print("🔍 Analyzing patterns...")
        
        # Detect anomalies
        anomaly_scores = self.detect_anomalies(all_features)
        
        # Semantic clustering
        embeddings_array = np.array(embeddings)
        clusters, transitions = self.cluster_semantic_moments(embeddings_array)
        
        # Score segments
        scored_segments = []
        for i, (segment, features) in enumerate(zip(segments, all_features)):
            if not features:
                continue
                
            score = 0
            
            # Scoring logic (same as before)
            if i < len(anomaly_scores):
                score += anomaly_scores[i] * 2.0
            
            if i in transitions:
                score += 3.0
            
            energy_mean = np.mean([f.get('audio_energy_mean', 0) for f in all_features])
            if features.get('audio_energy_max', 0) > energy_mean * 1.5:
                score += 2.5
            
            motion_mean = np.mean([f.get('motion_mean', 0) for f in all_features])
            if features.get('motion_max', 0) > motion_mean * 2:
                score += 2.0
            
            if features.get('face_count_max', 0) > 0:
                score += 1.5
            
            score += features.get('question_ratio', 0) * 10
            score += features.get('exclamation_ratio', 0) * 8
            score += features.get('scene_changes', 0) * 0.5
            
            duration = segment['end'] - segment['start']
            if 7 <= duration <= 12:
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
    def score_segments(self, segments, audio_path=None, video_path=None, 
                      processing_mode='auto', max_workers=None, batch_size=32):
        """
        Score segments with multiple processing modes
        
        processing_mode options:
        - 'auto': Automatically choose best method based on system resources
        - 'gpu': Use GPU acceleration with batching (recommended for 100+ segments)
        - 'memory_safe': Conservative CPU processing (safe for any system)
        - 'sequential': Original sequential processing
        """
        
        if not segments:
            return []
        
        segment_count = len(segments)
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        has_gpu = torch.cuda.is_available()
        
        print(f"📊 Processing {segment_count} segments")
        print(f"💾 Available memory: {available_memory_gb:.1f} GB")
        print(f"🎮 GPU available: {has_gpu}")
        
        # Auto-select processing mode
        if processing_mode == 'auto':
            if has_gpu and segment_count >= 100 and available_memory_gb >= 6:
                processing_mode = 'gpu'
            elif segment_count >= 50 and available_memory_gb >= 4:
                processing_mode = 'memory_safe'
            else:
                processing_mode = 'sequential'
        
        print(f"🚀 Using {processing_mode} processing mode")
        
        # Route to appropriate processing method
        if processing_mode == 'gpu' and has_gpu:
            return self.score_segments_gpu_batched(segments, audio_path, video_path, batch_size, max_workers or 2)
        elif processing_mode == 'memory_safe':
            return self.score_segments_memory_safe(segments, audio_path, video_path, max_workers)
        else:
            # Sequential processing (original safe method)
            return self._score_segments_sequential(segments, audio_path, video_path)
    
    def _score_segments_sequential(self, segments, audio_path=None, video_path=None):
        """Original sequential processing method (safest)"""
        print(f"🐌 Sequential processing {len(segments)} segments...")
        
        all_features = []
        embeddings = []
        
        for i, segment in enumerate(tqdm(segments, desc="Processing segments")):
            if i % 50 == 0:  # Memory check every 50 segments
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 85:
                    print(f"⚠️ High memory usage ({memory_percent:.1f}%), forcing cleanup...")
                    gc.collect()
                    time.sleep(1)
            
            try:
                features = self.extract_multimodal_features(segment, audio_path, video_path)
                all_features.append(features)
                embeddings.append(features['text_embedding'])
            except Exception as e:
                print(f"Error processing segment {i}: {e}")
                all_features.append(self._get_default_features())
                embeddings.append(np.zeros(384))
        
        return self._score_from_features(segments, all_features, embeddings)
    
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
                     num_shorts=5, context_seconds=2, processing_mode='auto', 
                     max_workers=None, batch_size=32):
        """Create viral shorts with crash-safe processing for large datasets"""
        
        if not whisper_segments:
            print("No segments provided")
            return []
        
        # Convert segments to dict format
        segments = []
        for seg in whisper_segments:
            try:
                if hasattr(seg, 'start'):
                    segments.append({
                        "start": float(seg.start),
                        "end": float(seg.end),
                        "text": str(seg.text)
                    })
                else:
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
        
        # Score segments using crash-safe ML approach
        scored_segments = self.detector.score_segments(
            segments, audio_path, video_path, 
            processing_mode=processing_mode,
            max_workers=max_workers,
            batch_size=batch_size
        )
        
        if not scored_segments:
            print("No scored segments found")
            return []
        
        # Select best highlights
        highlights = self.detector.select_diverse_highlights(scored_segments, top_n=num_shorts)
        
        # Create shorts with context
        shorts = []
        for highlight in highlights:
            start_time = max(0, highlight['start'] - context_seconds)
            end_time = highlight['end'] + context_seconds
            
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


# Example usage with crash-safe processing
def create_viral_shorts_ml(whisper_segments, audio_path=None, video_path=None, 
                          processing_mode='auto', max_workers=None, batch_size=32):
    """Main function with crash-safe processing for large datasets"""
    
    creator = ViralShortsCreator()
    
    # System resource check
    available_memory = psutil.virtual_memory().available / (1024**3)
    total_segments = len(whisper_segments)
    
    print(f"🖥️ System Status:")
    print(f"   Memory available: {available_memory:.1f} GB")
    print(f"   Segments to process: {total_segments}")
    print(f"   GPU available: {torch.cuda.is_available()}")
    
    # Adjust parameters based on system resources
    if available_memory < 4 and total_segments > 100:
        print("⚠️ Warning: Large dataset with limited memory detected")
        print("   Forcing memory-safe mode...")
        processing_mode = 'memory_safe'
        batch_size = min(batch_size, 16)
    
    # Create shorts with crash-safe processing
    shorts = creator.create_shorts(
        whisper_segments=whisper_segments,
        audio_path=audio_path,
        video_path=video_path,
        num_shorts=5,
        context_seconds=2,
        processing_mode=processing_mode,
        max_workers=max_workers,
        batch_size=batch_size
    )
    
    print(f"\n🎯 Found {len(shorts)} viral short opportunities:")
    for i, short in enumerate(shorts, 1):
        print(f"\n📹 Short #{i}:")
        print(f"   Duration: {short['duration']:.1f}s")
        print(f"   Score: {short['score']:.2f}")
        print(f"   Time: {short['start']:.1f}s - {short['end']:.1f}s")
        
        features = short['features']
        if features.get('audio_energy_max', 0) > 0:
            print(f"   🔊 High audio energy: {features['audio_energy_max']:.2f}")
        if features.get('motion_max', 0) > 0:
            print(f"   🎬 High visual motion: {features['motion_max']:.2f}")
        if features.get('face_count_max', 0) > 0:
            print(f"   👤 Human presence detected")
        
        print(f"   📝 Text: {short['highlight_text'][:80]}...")
    
    return shorts


# Safe usage examples for large datasets
if __name__ == "__main__":
    print("=" * 80)
    print("CRASH-SAFE PROCESSING FOR LARGE DATASETS")
    print("=" * 80)
    
    # Simulate 700 segments
    large_segments = []
    for i in range(700):
        large_segments.append({
            "start": i * 5,
            "end": i * 5 + 4,
            "text": f"Segment {i}: This could be viral content with emotional impact"
        })
    
    # Method 1: Auto mode (recommended - system chooses best approach)
    print("\n🤖 Method 1: Auto Mode (Recommended)")
    print("-" * 50)
    start_time = time.time()
    
    shorts_auto = create_viral_shorts_ml(
        whisper_segments=large_segments,
        audio_path="video.mp3",
        video_path="video.mp4",
        processing_mode='auto'  # System automatically chooses best method
    )
    
    auto_time = time.time() - start_time
    print(f"✅ Completed in {auto_time:.1f}s without crashing!")
    
    # Method 2: GPU mode (for high-end systems)
    if torch.cuda.is_available():
        print("\n🚀 Method 2: GPU Accelerated")
        print("-" * 50)
        start_time = time.time()
        
        shorts_gpu = create_viral_shorts_ml(
            whisper_segments=large_segments,
            audio_path="video.mp3", 
            video_path="video.mp4",
            processing_mode='gpu',
            batch_size=16,  # Smaller batches to prevent GPU memory issues
            max_workers=2   # Limited workers to prevent overload
        )
        
        gpu_time = time.time() - start_time
        print(f"✅ GPU processing completed in {gpu_time:.1f}s")
    else:
        print("\n❌ GPU not available, skipping GPU test")
    
    # Method 3: Memory-safe mode (for any system)
    print("\n🛡️ Method 3: Memory-Safe Mode (Universal)")
    print("-" * 50)
    start_time = time.time()
    
    shorts_safe = create_viral_shorts_ml(
        whisper_segments=large_segments,
        audio_path="video.mp3",
        video_path="video.mp4", 
        processing_mode='memory_safe',
        max_workers=2  # Conservative worker count
    )
    
    safe_time = time.time() - start_time
    print(f"✅ Memory-safe processing completed in {safe_time:.1f}s")
    
    # Method 4: For systems with very limited resources
    print("\n🐌 Method 4: Ultra-Safe Sequential (Low-end systems)")
    print("-" * 50)
    start_time = time.time()
    
    # Test with smaller dataset for demo
    small_segments = large_segments[:50]
    shorts_sequential = create_viral_shorts_ml(
        whisper_segments=small_segments,
        audio_path="video.mp3",
        video_path="video.mp4",
        processing_mode='sequential'
    )
    
    sequential_time = time.time() - start_time
    estimated_full_time = sequential_time * (700/50)
    print(f"✅ Sequential (50 segments) completed in {sequential_time:.1f}s")
    print(f"📊 Estimated time for 700 segments: {estimated_full_time:.1f}s")
    
    print("\n" + "=" * 80)
    print("🎉 ALL PROCESSING MODES COMPLETED WITHOUT CRASHING!")
    print("💡 Tips for your 700 segments:")
    print("   • Use 'auto' mode for best performance")
    print("   • Use 'memory_safe' if system has limited RAM")
    print("   • Use 'gpu' mode if you have a dedicated GPU")
    print("   • Monitor system resources during processing")
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