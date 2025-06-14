import os
import os
import pickle
import hashlib
from faster_whisper import WhisperModel



TRANSCRIPTION_CACHE_DIR = ".transcription_cache"

def get_video_checksum(file_path):
    """Create unique hash based on file contents and metadata"""
    file_stat = os.stat(file_path)
    hash_data = f"{file_stat.st_size}-{file_stat.st_mtime_ns}"
    return hashlib.md5(hash_data.encode()).hexdigest()

def get_transcription(force_refresh=False, model_version="base", input_file=None):
    # Create cache directory if needed
    os.makedirs(TRANSCRIPTION_CACHE_DIR, exist_ok=True)
    
    cache_key = f"{get_video_checksum(input_file)}_{model_version}"
    cache_path = os.path.join(TRANSCRIPTION_CACHE_DIR, f"{cache_key}.pkl")
    
    # Return cached result if available
    if not force_refresh and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            print(f"Loading cached transcription from {cache_path}")
            return pickle.load(f)
    
    # Generate new transcription
    #model = whisper.load_model(model_version, device="cuda")
    #TODO fix cuda using cpu for now
    model = WhisperModel(model_version)
    segments, info = model.transcribe(input_file)
    
    segments_list = list(segments)
    # Cache the results
    with open(cache_path, "wb") as f:
        pickle.dump(segments_list, f) 
        print(f"Cached transcription to {cache_path}")
    
    return segments_list