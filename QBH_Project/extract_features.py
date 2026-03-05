"""
extract_features.py — Fast Feature Extraction for QBH

Key decisions:
  - intervals[::3] (every 3rd frame) — aggressive downsampling, keeps melody shape, very fast DTW
  - Z-score on full sequence before downsampling
  - No onset features — unreliable from mic recordings
  - Chroma = lightweight pre-filter tie-breaker only
  - Database is only built once (skip if pkl exists), use --force to rebuild
"""

import os
import sys
import numpy as np
import librosa
import pickle
from scipy.ndimage import median_filter

PROJECT_DIR  = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR   = os.path.dirname(PROJECT_DIR)
WAV_DIR      = os.path.join(PROJECT_DIR, "wav_songs")
FEATURES_PKL = os.path.join(PROJECT_DIR, "database", "saved_features.pkl")


def _pitch_stability_filter(f0, min_run=3):
    """Remove isolated voiced frames (runs shorter than min_run)."""
    voiced = ~np.isnan(f0) & (f0 > 0)
    clean  = np.copy(f0)
    i = 0
    while i < len(voiced):
        if voiced[i]:
            j = i
            while j < len(voiced) and voiced[j]:
                j += 1
            if (j - i) < min_run:
                clean[i:j] = np.nan
            i = j
        else:
            i += 1
    return clean


def extract_features(file_path, hop_length=512):
    """
    Extract pitch interval contour + normalized chroma.

    Returns (intervals, chroma, None).
    intervals: every-3rd-frame subsampled, z-score normalized, clamped
    chroma:    mean L2-normalized 12-dim vector
    """
    try:
        sr = 22050

        # Load + normalize
        y, sr = librosa.load(file_path, sr=sr, mono=True)
        y = librosa.util.normalize(y)

        # Gentle trim — top_db=30 preserves soft humming
        y, _ = librosa.effects.trim(y, top_db=30)
        if len(y) < sr:
            return None, None, None

        # Harmonic separation
        y_harmonic, _ = librosa.effects.hpss(y)

        # pyin pitch
        f0, _, _ = librosa.pyin(
            y_harmonic,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            hop_length=hop_length,
            fill_na=np.nan
        )

        # Stability filter — kill isolated pops
        f0 = _pitch_stability_filter(f0, min_run=3)
        f0_voiced = f0[~np.isnan(f0) & (f0 > 0)]

        if len(f0_voiced) < 10:
            return None, None, None

        # Hz → MIDI → intervals
        midi      = 69 + 12 * np.log2(f0_voiced / 440.0)
        intervals = np.diff(midi)

        if len(intervals) < 5:
            return None, None, None

        # Semitone quantization (key-invariant, pitch-error tolerant)
        intervals = np.round(intervals)

        # Light smoothing
        intervals = median_filter(intervals, size=3)

        # Z-score on FULL sequence (before any downsampling)
        iv_std = np.std(intervals)
        if iv_std > 0:
            intervals = (intervals - np.mean(intervals)) / iv_std

        # Aggressive downsampling: every 3rd frame
        # Preserves melody shape, drastically reduces DTW computation
        intervals = intervals[::3]

        # Hard cap at 300 — prevents DTW explosion on long songs
        # 300 frames ≈ first ~30 seconds of melody (most identifiable part)
        if len(intervals) > 300:
            intervals = intervals[:300]

        # Clamp outliers to [-3, 3] — reduces influence of extreme jumps
        intervals = np.clip(intervals, -3.0, 3.0)

        # Chroma (pre-filter tie-breaker)
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=hop_length)
        norms  = np.linalg.norm(chroma, axis=0, keepdims=True)
        norms[norms == 0] = 1
        chroma_mean = np.mean(chroma / norms, axis=1)
        cn = np.linalg.norm(chroma_mean)
        if cn > 0:
            chroma_mean = chroma_mean / cn

        return (
            intervals.astype(np.float32),
            chroma_mean.astype(np.float32),
            None
        )

    except Exception as e:
        print(f"  ERR  {os.path.basename(file_path)}: {e}")
        import traceback; traceback.print_exc()
        return None, None, None


def build_feature_database(wav_dir=WAV_DIR, pkl_path=FEATURES_PKL, force=False):
    """
    Extract features for all WAV files and save to saved_features.pkl.
    Skips extraction if pkl already exists (pass --force on CLI or force=True to rebuild).
    """
    if os.path.exists(pkl_path) and not force:
        print(f"[features] Database already exists → '{pkl_path}'. Skipping.")
        print(f"[features] Run with --force to rebuild.")
        return

    if not os.path.exists(wav_dir):
        print(f"Error: '{wav_dir}' not found. Run convert.py first.")
        return

    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)

    files = sorted(f for f in os.listdir(wav_dir) if f.endswith('.wav'))
    if not files:
        print(f"No WAV files in '{wav_dir}'. Run convert.py first.")
        return

    print(f"Extracting features from {len(files)} songs…\n")
    db = {}
    for fn in files:
        fp = os.path.join(wav_dir, fn)
        intervals, chroma, _ = extract_features(fp)
        if intervals is not None:
            db[fn] = {"intervals": intervals, "chroma": chroma}
            print(f"  OK   {fn}  (intervals={len(intervals)}, chroma={chroma.shape})")
        else:
            print(f"  SKIP {fn} (insufficient pitch data)")

    with open(pkl_path, 'wb') as f:
        pickle.dump(db, f)
    print(f"\nDone! {len(db)} songs saved to '{pkl_path}'.")


if __name__ == "__main__":
    force = "--force" in sys.argv
    build_feature_database(force=force)
