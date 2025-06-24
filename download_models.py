#!/usr/bin/env python3
"""
Download all required models for the viral clip generator.
This script pre-downloads all models to avoid download delays during processing.
"""

import os
import sys
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import config

console = Console()

def download_transformers_models():
    """Download all Transformers models to local models/ directory."""
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
        pipeline, CLIPProcessor, CLIPModel
    )
    
    models_to_download = [
        {
            'name': 'Emotion Detection Model',
            'model_id': 'j-hartmann/emotion-english-distilroberta-base',
            'local_path': 'emotion-model',
            'type': 'classification'
        },
        {
            'name': 'CLIP Vision Model',
            'model_id': 'openai/clip-vit-base-patch32',
            'local_path': 'clip-model',
            'type': 'clip',
        },
        {
            'name': 'Sentiment Analysis Model',
            'model_id': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
            'local_path': 'sentiment-model',
            'type': 'classification'
        },
        {
            'name': 'Main LLM (DialoGPT)',
            'model_id': 'microsoft/DialoGPT-medium',
            'local_path': 'dialogpt-model',
            'type': 'causal_lm'
        }
    ]
    
    console.print("[blue]📥 Downloading Transformers models to local directory...[/blue]")
    
    for model_info in models_to_download:
        try:
            console.print(f"[cyan]Downloading {model_info['name']}...[/cyan]")
            local_model_path = config.MODELS_DIR / model_info['local_path']
            local_model_path.mkdir(parents=True, exist_ok=True)
            
            if model_info['type'] == 'classification':
                # Download tokenizer and model for classification
                tokenizer = AutoTokenizer.from_pretrained(model_info['model_id'])
                model = AutoModelForSequenceClassification.from_pretrained(model_info['model_id'])
                
                # Save locally
                tokenizer.save_pretrained(local_model_path)
                model.save_pretrained(local_model_path)
                console.print(f"[green]✓ {model_info['name']} saved to {local_model_path}[/green]")
                
            elif model_info['type'] == 'clip':
                # Download CLIP model and processor
                model = CLIPModel.from_pretrained(model_info['model_id'])
                processor = CLIPProcessor.from_pretrained(model_info['model_id'])
                
                # Save locally
                model.save_pretrained(local_model_path)
                processor.save_pretrained(local_model_path)
                console.print(f"[green]✓ {model_info['name']} saved to {local_model_path}[/green]")
                
            elif model_info['type'] == 'causal_lm':
                # Download tokenizer and model separately
                tokenizer = AutoTokenizer.from_pretrained(model_info['model_id'])
                model = AutoModelForCausalLM.from_pretrained(
                    model_info['model_id'],
                    torch_dtype='auto',
                    low_cpu_mem_usage=True
                )
                
                # Save locally
                tokenizer.save_pretrained(local_model_path)
                model.save_pretrained(local_model_path)
                console.print(f"[green]✓ {model_info['name']} saved to {local_model_path}[/green]")
                
        except Exception as e:
            console.print(f"[red]✗ Failed to download {model_info['name']}: {e}[/red]")
            continue

def download_sentence_transformers():
    """Download sentence transformer models to local directory."""
    try:
        from sentence_transformers import SentenceTransformer
        
        console.print("[cyan]Downloading Sentence Transformer model...[/cyan]")
        local_model_path = config.MODELS_DIR / "sentence-transformer"
        local_model_path.mkdir(parents=True, exist_ok=True)
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        model.save(str(local_model_path))
        console.print(f"[green]✓ Sentence Transformer saved to {local_model_path}[/green]")
        
    except Exception as e:
        console.print(f"[red]✗ Failed to download Sentence Transformer: {e}[/red]")

def download_whisper_model():
    """Download Whisper model to local directory."""
    try:
        import whisper
        import shutil
        
        console.print(f"[cyan]Downloading Whisper model: {config.WHISPER_MODEL}...[/cyan]")
        
        # Load model (this downloads to cache first)
        model = whisper.load_model(config.WHISPER_MODEL, device="cpu")
        
        # Copy from cache to local models directory
        whisper_cache_dir = Path.home() / ".cache" / "whisper"
        local_whisper_dir = config.MODELS_DIR / "whisper"
        local_whisper_dir.mkdir(parents=True, exist_ok=True)
        
        # Find the model file in cache
        model_file = f"{config.WHISPER_MODEL}.pt"
        cache_model_path = whisper_cache_dir / model_file
        local_model_path = local_whisper_dir / model_file
        
        if cache_model_path.exists():
            shutil.copy2(cache_model_path, local_model_path)
            console.print(f"[green]✓ Whisper model saved to {local_model_path}[/green]")
        else:
            console.print("[green]✓ Whisper model downloaded successfully[/green]")
        
    except Exception as e:
        console.print(f"[red]✗ Failed to download Whisper model: {e}[/red]")

def check_yolo_model():
    """Check if YOLO model exists."""
    yolo_path = Path(config.YOLO_MODEL)
    if yolo_path.exists():
        console.print(f"[green]✓ YOLO model found: {yolo_path}[/green]")
        return True
    else:
        console.print(f"[red]✗ YOLO model not found: {yolo_path}[/red]")
        console.print("[yellow]Please ensure yolov8n.pt is in the models/ directory[/yellow]")
        return False

def main():
    """Main function to download all models."""
    console.print("""
[bold blue]🤖 AI Model Downloader for Viral Clip Generator[/bold blue]
═══════════════════════════════════════════════════

This script will download all required AI models:
• Whisper (Speech-to-Text)
• YOLO (Object Detection) 
• CLIP (Vision Understanding)
• Emotion Detection
• Sentiment Analysis
• Sentence Transformers
• Language Models

[yellow]Note: This may take several minutes and require significant disk space.[/yellow]
""")
    
    # Create models directory
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # Check YOLO model first
        yolo_ok = check_yolo_model()
        
        # Download Whisper model
        download_whisper_model()
        
        # Download Transformers models
        download_transformers_models()
        
        # Download Sentence Transformers
        download_sentence_transformers()
        
        # Summary
        console.print(f"""
[green]🎉 Model download process completed![/green]

[cyan]Models are stored locally in:[/cyan]
• Transformers models: {config.MODELS_DIR}/
• Whisper models: {config.MODELS_DIR}/whisper/
• Sentence Transformers: {config.MODELS_DIR}/sentence-transformer/
• YOLO model: {config.MODELS_DIR}/yolov8n.pt

[yellow]Next steps:[/yellow]
1. {"✓" if yolo_ok else "✗"} Ensure YOLO model (yolov8n.pt) is in models/ directory
2. Test the pipeline with: python main.py --help
3. Generate your first viral clip!
""")
        
        return 0 if yolo_ok else 1
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Download interrupted by user[/yellow]")
        return 1
    except Exception as e:
        console.print(f"\n[red]💥 Unexpected error: {e}[/red]")
        return 1

if __name__ == "__main__":
    sys.exit(main())