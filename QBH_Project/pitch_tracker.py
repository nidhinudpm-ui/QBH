import librosa
import numpy as np
from scipy.signal import medfilt

# ─────────────────────────────────────────────────────────────────────────────
#  pYIN BACKEND (unchanged baseline)
# ─────────────────────────────────────────────────────────────────────────────

def extract_f0_pyin(y, sr, fmin, fmax, hop_length):
    """Extract continuous F0 using pYIN algorithm."""
    if isinstance(fmin, str):
        fmin = librosa.note_to_hz(fmin)
    if isinstance(fmax, str):
        fmax = librosa.note_to_hz(fmax)

    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr,
        hop_length=hop_length, fill_na=0.0
    )
    return f0


def interpolate_small_gaps(f0, max_gap=10):
    """Fill in small zero gaps in the F0 contour by linear interpolation."""
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
    """Convert Hz to MIDI-style semitones, preserving 0 for unvoiced frames."""
    semitones = np.zeros_like(f0)
    mask = f0 > 0
    semitones[mask] = 12 * np.log2(f0[mask] / 440.0) + 69
    return semitones


def smooth_f0(f0, kernel_size=5):
    """Apply median filter to reduce pitch jitter."""
    if len(f0) < kernel_size:
        return f0
    return medfilt(f0, kernel_size)


def get_continuous_contour(y, sr, fmin, fmax, hop_length):
    """Full pYIN pipeline: extract → interpolate gaps → smooth → semitones."""
    f0 = extract_f0_pyin(y, sr, fmin, fmax, hop_length)
    f0 = interpolate_small_gaps(f0)
    f0 = smooth_f0(f0)
    semitones = hz_to_semitones(f0)
    return semitones


# ─────────────────────────────────────────────────────────────────────────────
#  torchcrepe BACKEND (query-only experimental)
# ─────────────────────────────────────────────────────────────────────────────

def get_continuous_contour_torchcrepe(y, sr):
    """
    Extract F0 semitone contour using torchcrepe.
    Returns a float32 semitone array compatible with the pYIN path.
    Raises RuntimeError if torchcrepe is unavailable or prediction fails.
    """
    import importlib
    if importlib.util.find_spec("torchcrepe") is None:
        raise RuntimeError("torchcrepe is not installed")

    import torch
    import torchcrepe
    from config import (
        TORCHCREPE_SAMPLE_RATE, TORCHCREPE_HOP_LENGTH, TORCHCREPE_MODEL,
        TORCHCREPE_FMIN, TORCHCREPE_FMAX, TORCHCREPE_BATCH_SIZE,
        TORCHCREPE_VOICING_THRESHOLD, TORCHCREPE_USE_GPU_IF_AVAILABLE,
        MIN_VOICED_FRAMES, F0_SMOOTH_KERNEL
    )

    # Resample if needed
    if sr != TORCHCREPE_SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=TORCHCREPE_SAMPLE_RATE)
        sr = TORCHCREPE_SAMPLE_RATE

    # Normalize safely
    max_val = np.abs(y).max()
    if max_val > 0:
        y = y / max_val

    # Device selection
    device = "cpu"
    if TORCHCREPE_USE_GPU_IF_AVAILABLE and torch.cuda.is_available():
        device = "cuda"

    # torchcrepe expects [1, N] float32 tensor
    audio_tensor = torch.tensor(y[np.newaxis, :], dtype=torch.float32).to(device)

    # Run prediction: returns (frequency, periodicity) tensors
    frequency, periodicity = torchcrepe.predict(
        audio_tensor,
        sample_rate=sr,
        hop_length=TORCHCREPE_HOP_LENGTH,
        fmin=TORCHCREPE_FMIN,
        fmax=TORCHCREPE_FMAX,
        model=TORCHCREPE_MODEL,
        batch_size=TORCHCREPE_BATCH_SIZE,
        decoder=torchcrepe.decode.weighted_argmax,
        return_periodicity=True,
        device=device,
    )

    freq_np = frequency.squeeze().cpu().numpy()   # shape: (T,)
    period_np = periodicity.squeeze().cpu().numpy()  # confidence

    # Suppress low-confidence frames
    freq_np = freq_np.copy()
    freq_np[period_np < TORCHCREPE_VOICING_THRESHOLD] = 0.0

    # --- Voiced-frame guard ---
    voiced_count = np.sum(freq_np > 0)
    if voiced_count < MIN_VOICED_FRAMES:
        raise RuntimeError(
            f"torchcrepe: too few confident voiced frames ({voiced_count} < {MIN_VOICED_FRAMES})"
        )

    # --- Shared post-processing ---
    freq_np = interpolate_small_gaps(freq_np, max_gap=10)
    freq_np = smooth_f0(freq_np, kernel_size=F0_SMOOTH_KERNEL)
    semitones = hz_to_semitones(freq_np)

    # Normalize by voiced median (backend-agnostic)
    voiced_mask = semitones > 0
    if voiced_mask.any():
        semitones[voiced_mask] -= np.median(semitones[voiced_mask])

    return semitones.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────

def get_query_contour(y, sr, backend="pyin"):
    """
    Dispatcher for query pitch extraction.

    Returns a dict:
      {
        "contour":       np.ndarray (float32 semitones),
        "backend_used":  "pyin" | "torchcrepe",
        "fallback_used": bool
      }

    If torchcrepe fails (import error, too few voiced frames, any exception),
    falls back to pYIN automatically with a printed warning.
    """
    from config import PYIN_FMIN, PYIN_FMAX, F0_HOP_LENGTH

    if backend not in ("pyin", "torchcrepe"):
        raise ValueError(f"Unknown pitch backend: '{backend}'. Use 'pyin' or 'torchcrepe'.")

    if backend == "torchcrepe":
        try:
            contour = get_continuous_contour_torchcrepe(y, sr)
            print(
                f"[pitch_tracker] Backend=torchcrepe | frames={len(contour)} | fallback=False",
                flush=True
            )
            return {"contour": contour, "backend_used": "torchcrepe", "fallback_used": False}
        except Exception as e:
            print(
                f"[pitch_tracker] WARNING: torchcrepe failed ({e}). Falling back to pYIN.",
                flush=True
            )
            # Fall through to pYIN below

    # pYIN path (baseline or fallback)
    contour = get_continuous_contour(y, sr, PYIN_FMIN, PYIN_FMAX, F0_HOP_LENGTH)
    fallback = (backend == "torchcrepe")  # True only if we were asked for torchcrepe but fell back
    print(
        f"[pitch_tracker] Backend=pyin | frames={len(contour)} | fallback={fallback}",
        flush=True
    )
    return {"contour": contour, "backend_used": "pyin", "fallback_used": fallback}
