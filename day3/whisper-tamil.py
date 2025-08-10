# pip install torch transformers librosa gradio soundfile resampy


import argparse
import os
import sys
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import gradio as gr

MODEL_ID = "vasista22/whisper-tamil-small"

def download_model():
    """Download the model and return processor and model objects."""
    print(f"Loading model: {MODEL_ID}...")
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("Using GPU for inference")
        model = model.to(device)
    else:
        print("Using CPU for inference (consider using GPU for faster processing)")
    
    return processor, model, device

def transcribe_audio(audio_file, processor, model, device):
    """Transcribe Tamil audio file using the loaded model."""
    import numpy as np
    
    print(f"Transcribing: {audio_file}")
    
    # Load audio directly with librosa (more reliable than datasets)
    try:
        import librosa
        
        # Load audio with librosa (ensures correct handling of various audio formats)
        audio_array, original_sampling_rate = librosa.load(audio_file, sr=None)
        
        # Resample to 16000 Hz which is required by Whisper
        if original_sampling_rate != 16000:
            print(f"Resampling audio from {original_sampling_rate}Hz to 16000Hz")
            audio_array = librosa.resample(
                audio_array, 
                orig_sr=original_sampling_rate, 
                target_sr=16000
            )
        sampling_rate = 16000  # Whisper requires 16kHz sampling rate
        
    except Exception as e:
        print(f"Error loading audio with librosa: {e}")
        # Try alternative loading method with soundfile
        try:
            import soundfile as sf
            audio_array, original_sampling_rate = sf.read(audio_file)
            
            # Convert stereo to mono if needed
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
                
            # Resample to 16000 Hz if needed
            if original_sampling_rate != 16000:
                print(f"Resampling audio from {original_sampling_rate}Hz to 16000Hz")
                import resampy
                audio_array = resampy.resample(
                    audio_array, 
                    original_sampling_rate, 
                    16000
                )
            sampling_rate = 16000
            
        except Exception as e2:
            print(f"Failed to load audio with soundfile: {e2}")
            return f"Error: Could not load audio file. Please ensure the file is a valid audio format."
    
    # Process audio with the fixed sampling rate
    try:
        input_features = processor(
            audio_array, 
            sampling_rate=16000,  # Always use 16000 as the sampling rate for Whisper
            return_tensors="pt"
        ).input_features
        
        if device == "cuda":
            input_features = input_features.to(device)
        
        # Generate tokens
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        
        # Decode the tokens to text
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return transcription
    except Exception as e:
        print(f"Error during processing or transcription: {e}")
        return f"Error during processing: {str(e)}"

def create_web_ui(processor, model, device):
    """Create a Gradio web UI for the ASR system."""
    def transcribe_audio_ui(audio_path):
        if audio_path is None:
            return "No audio file provided"
        
        result = transcribe_audio(audio_path, processor, model, device)
        return result
    
    ui = gr.Interface(
        fn=transcribe_audio_ui,
        inputs=gr.Audio(type="filepath", label="Upload Tamil Audio"),
        outputs=gr.Textbox(label="Tamil Transcription"),
        title="Tamil Speech Recognition",
        description="Transcribe Tamil speech to text using vasista22/whisper-tamil-small model",
        examples=[],
        allow_flagging="never"
    )
    
    return ui

def main():
    """Main function to run the ASR web UI."""
    parser = argparse.ArgumentParser(description="Tamil Automatic Speech Recognition Web UI")
    parser.add_argument("--port", type=int, default=7860, help="Port for web UI")
    
    args = parser.parse_args()

    # Download model
    processor, model, device = download_model()
    
    # Launch web UI
    ui = create_web_ui(processor, model, device)
    ui.launch(server_port=args.port)

if __name__ == "__main__":
    main()