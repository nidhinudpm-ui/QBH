
import librosa
import numpy as np
from scipy.signal import medfilt

def extract_f0_pyin(y, sr, fmin, fmax, hop_length):
    if isinstance(fmin, str):
        fmin = librosa.note_to_hz(fmin)
    if isinstance(fmax, str):
        fmax = librosa.note_to_hz(fmax)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
    return np.nan_to_num(f0, nan=0.0)

def interpolate_small_gaps(f0, max_gap=10):
    f0_interp = f0.copy()
    n = len(f0)
    i = 0
    while i < n:
        if f0[i] == 0:
            start = i
            while i < n and f0[i] == 0:
                i += 1
            end = i
            if 0 < start and end < n and (end - start) <= max_gap:
                v_start = f0[start - 1]
                v_end = f0[end]
                f0_interp[start:end] = np.linspace(v_start, v_end, end - start + 2)[1:-1]
        else:
            i += 1
    return f0_interp

def hz_to_semitones(f0):
    semitones = np.zeros_like(f0)
    mask = f0 > 0
    semitones[mask] = 12 * np.log2(f0[mask] / 440.0) + 69
    return semitones

def smooth_f0(f0, kernel_size=5):
    if len(f0) < kernel_size: return f0
    return medfilt(f0, kernel_size)

def get_continuous_contour(y, sr, fmin, fmax, hop_length):
    f0 = extract_f0_pyin(y, sr, fmin, fmax, hop_length)
    f0 = interpolate_small_gaps(f0)
    f0 = smooth_f0(f0)
    semitones = hz_to_semitones(f0)
    voiced_mask = semitones > 0
    if voiced_mask.any():
        semitones[voiced_mask] -= np.median(semitones[voiced_mask])
    return semitones.astype(np.float32)

def get_query_contour(y, sr, backend="pyin"):
    # Force pYIN for now to bypass torchcrepe issues
    from config import PYIN_FMIN, PYIN_FMAX, F0_HOP_LENGTH
    contour = get_continuous_contour(y, sr, PYIN_FMIN, PYIN_FMAX, F0_HOP_LENGTH)
    return {"contour": contour, "backend_used": "pyin", "fallback_used": False}
