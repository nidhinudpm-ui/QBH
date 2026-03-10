"""
audio_preprocess.py — Canonical audio preprocessing for QBH queries and songs.

Pipeline:
  1. High-pass filter (remove rumble below cutoff)
  2. Light noise reduction (stationary spectral gating)
  3. Spectral energy gating (trim low-energy frames)
  4. Silence trimming
  5. Peak normalization

NO pre-emphasis — it boosts high-frequency breathiness and
degrades pYIN pitch tracking on hums.
"""

import numpy as np
import librosa
from scipy.signal import butter, filtfilt, sosfilt

from config import HPF_CUTOFF, SAMPLE_RATE


# ─── Building Blocks ─────────────────────────────────────────────────────────

def highpass_filter(y, sr, cutoff=None):
    """Remove low-frequency rumble with a 5th-order Butterworth HPF."""
    cutoff = cutoff or HPF_CUTOFF
    sos = butter(5, cutoff, btype='highpass', fs=sr, output='sos')
    return sosfilt(sos, y).astype(np.float32)


def normalize_audio(y):
    """Peak normalization to [-1, 1]."""
    peak = np.max(np.abs(y))
    if peak > 0:
        return (y / peak).astype(np.float32)
    return y


def trim_silence(y, sr, top_db=20):
    """Trim leading/trailing silence."""
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return trimmed


def reduce_noise_light(y, sr):
    """
    Light stationary noise reduction via noisereduce.
    prop_decrease=0.6 is gentler than 0.7 to preserve pitch detail.
    """
    try:
        import noisereduce as nr
        return nr.reduce_noise(y=y, sr=sr, prop_decrease=0.6, stationary=True).astype(np.float32)
    except ImportError:
        # noisereduce is optional; skip if not installed
        return y


def spectral_energy_gate(y, sr, hop_length=512, threshold_ratio=0.3):
    """
    Trim frames whose spectral energy is below a fraction of the mean.
    Removes breathing noise and low-energy silence gaps.
    """
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    energy = np.sum(S, axis=0)
    threshold = np.mean(energy) * threshold_ratio

    mask = energy > threshold
    if not np.any(mask):
        return y  # Nothing to trim

    frames = np.where(mask)[0]
    start_sample = frames[0] * hop_length
    end_sample = min(frames[-1] * hop_length + hop_length, len(y))
    return y[start_sample:end_sample]


# ─── Full Pipelines ──────────────────────────────────────────────────────────

def preprocess_query_audio(y, sr):
    """
    Standard preprocessing for user recordings (hum/sing from browser mic).
    
    Order:
      1. High-pass filter at HPF_CUTOFF Hz
      2. Light noise reduction
      3. Spectral energy gating
      4. Silence trimming
      5. Peak normalization
    """
    y = highpass_filter(y, sr)
    y = reduce_noise_light(y, sr)
    y = spectral_energy_gate(y, sr)
    y = trim_silence(y, sr)
    y = normalize_audio(y)
    return y


def preprocess_song_audio(y, sr):
    """
    Preprocessing for library songs — lighter touch since songs are
    already mastered. Just normalize and trim silence.
    """
    y = trim_silence(y, sr)
    y = normalize_audio(y)
    return y
