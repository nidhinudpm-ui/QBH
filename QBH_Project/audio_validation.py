"""
audio_validation.py — Query validation and query-type detection.

Simplified first-pass classifier:
  - Voiced frame ratio (pYIN)
  - Pitch continuity (gap count)
  - Energy stability
  - Returns 'hum', 'mixed', or 'noise'
  (Separate 'sing' vs 'mixed' deferred until lyric branch is mature.)
"""

import numpy as np
import librosa

from config import SAMPLE_RATE, PYIN_FMIN, PYIN_FMAX


def is_valid_audio(audio, sr, min_rms=0.005, min_voiced_frames=15):
    """
    Basic validity check: energy + voiced content.
    Returns (is_valid, reason).
    """
    if len(audio) == 0:
        return False, "Empty audio"

    rms = np.sqrt(np.mean(audio ** 2))
    if rms < min_rms:
        return False, "No sound detected. Please hum into the microphone."

    # Voiced frame check via pYIN
    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz(PYIN_FMIN),
            fmax=librosa.note_to_hz(PYIN_FMAX),
            sr=sr,
            hop_length=512,
            fill_na=np.nan
        )
        voiced_count = np.sum(~np.isnan(f0) & (f0 > 0))
        if voiced_count < min_voiced_frames:
            return False, "No humming detected. Please hum a melody clearly."
    except Exception:
        pass  # If pYIN fails, let it through

    return True, "OK"


def detect_query_type(audio, sr):
    """
    Classify query as 'hum', 'mixed', or 'noise'.

    Heuristics:
      1. Voiced ratio — fraction of frames that are pitched.
         High voiced ratio + smooth pitch = hum.
      2. Pitch gap count — number of transitions from voiced→unvoiced.
         Frequent gaps suggest words/consonants = mixed/sing.
      3. Energy stability — std of frame energies.
         Stable energy = hum; dynamic energy = singing.

    Returns: 'hum', 'mixed', or 'noise'
    """
    # Step 1: Extract F0 for voiced analysis
    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz(PYIN_FMIN),
            fmax=librosa.note_to_hz(PYIN_FMAX),
            sr=sr,
            hop_length=512,
            fill_na=0.0
        )
    except Exception:
        return "mixed"  # Safe default

    if f0 is None or len(f0) == 0:
        return "noise"

    # Voiced ratio
    voiced_mask = f0 > 0
    voiced_ratio = np.mean(voiced_mask)

    if voiced_ratio < 0.10:
        return "noise"

    # Pitch gap count — transitions from voiced → unvoiced
    transitions = np.diff(voiced_mask.astype(int))
    gap_count = np.sum(transitions == -1)  # voiced → unvoiced edges
    frames_total = len(f0)
    gap_density = gap_count / max(frames_total, 1)

    # Energy stability — coefficient of variation of frame RMS
    frame_length = 1024
    hop_length = 512
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    rms_cv = np.std(rms) / (np.mean(rms) + 1e-8)

    # Classification thresholds
    # Hum: high voiced ratio, few gaps, stable energy
    # Mixed: lower voiced ratio or many gaps or dynamic energy
    if voiced_ratio > 0.55 and gap_density < 0.02 and rms_cv < 1.2:
        return "hum"
    else:
        return "mixed"
