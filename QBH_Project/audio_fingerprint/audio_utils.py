"""
audio_fingerprint/audio_utils.py — Utilities for audio conversion using librosa.
"""

import os
import librosa
import soundfile as sf
import numpy as np

def convert_to_wav(input_path: str, output_path: str, sample_rate: int = 11025):
    """
    Convert any uploaded audio file to mono WAV for audfprint using librosa.
    This replaces the ffmpeg dependency with a robust Python-only solution.
    """
    print(f"[audio_utils] Normalizing {input_path} to {output_path} at {sample_rate}Hz...")
    
    try:
        # Load with librosa (handles many formats via soundfile/audioread)
        # mono=True ensures 1 channel
        y, sr = librosa.load(input_path, sr=sample_rate, mono=True)
        
        # Save to WAV
        sf.write(output_path, y, sample_rate, subtype='PCM_16')
        
        print(f"[audio_utils] Success: {len(y)} samples saved.")
        return output_path
        
    except Exception as e:
        print(f"[audio_utils] Conversion failed: {str(e)}")
        raise RuntimeError(f"Audio normalization failed: {str(e)}")
