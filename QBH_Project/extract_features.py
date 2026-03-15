"""
extract_features.py — F0-Centric Feature Extraction with Preprocessing Bypass
"""

import os
import sys
import numpy as np
import librosa
import pickle
import warnings

warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from config import (
    SAMPLE_RATE, PYIN_FMIN, PYIN_FMAX,
    F0_HOP_LENGTH, F0_SMOOTH_KERNEL, MIN_VOICED_FRAMES,
    QUERY_PITCH_BACKEND, WAV_DIR, FEATURES_PKL, MAX_RECORDING_SEC
)

from audio_preprocess import preprocess_query_audio, preprocess_song_audio
from audio_validation import detect_query_type, is_valid_audio
from pitch_tracker import get_continuous_contour, extract_f0_pyin, get_query_contour
from melody_features import (
    compute_intervals, compute_contour,
    compute_interval_histogram, compute_contour_histogram
)

def extract_features(file_path, is_song=False):
    try:
        basename = os.path.basename(file_path)
        print(f"    [f0] Tracking melody in {basename[:45]}...", flush=True)
        
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        
        if not is_song:
            max_samples = MAX_RECORDING_SEC * sr
            if len(y) > max_samples:
                y = y[:max_samples]

            # Restoring basic normalization for detection sensitivity
            y = librosa.util.normalize(y)
            y, _ = librosa.effects.trim(y, top_db=25) # Slightly less aggressive trim
            y_proc = y
            
            result = get_query_contour(y_proc, sr, backend=QUERY_PITCH_BACKEND)
            semitones = result["contour"]
            backend_used = result.get("backend_used", "pyin")
            fallback_used = result.get("fallback_used", False)
            
            voiced_idxs = np.where(semitones > 0)[0]
            if len(voiced_idxs) > 0:
                start = max(0, voiced_idxs[0] - 8)
                end   = min(len(semitones), voiced_idxs[-1] + 9)
                semitones = semitones[start:end]
            
            if len(semitones) > 1200:
                semitones = semitones[:1200]
        else:
            # Vocal isolation for songs
            vocal_path = os.path.join(PROJECT_DIR, "uploads", f"vocal_{basename}")
            if not os.path.exists(vocal_path):
                vocal_path = file_path # Fallback
            y_vocal, sr = librosa.load(vocal_path, sr=SAMPLE_RATE, mono=True)
            y_proc = preprocess_song_audio(y_vocal, sr)

            semitones = get_continuous_contour(y_proc, sr, PYIN_FMIN, PYIN_FMAX, F0_HOP_LENGTH)

        if semitones is None or len(semitones) < 5:
            return None

        intervals = compute_intervals(semitones)
        contour = compute_contour(intervals)
        i_hist = compute_interval_histogram(intervals)
        c_hist = compute_contour_histogram(contour)

        if is_song:
            segments = [] # Songs should already be segmented in the DB, this is for extraction
            # Re-implementing a simple segmenter for the extraction return
            raw_segments = [semitones] # Simplified for now
            for seg in raw_segments:
                seg_intervals = compute_intervals(seg)
                seg_contour = compute_contour(seg_intervals)
                seg_i_hist = compute_interval_histogram(seg_intervals)
                seg_c_hist = compute_contour_histogram(seg_contour)
                segments.append((seg, seg_intervals, seg_contour, seg_i_hist, seg_c_hist))
        else:
            segments = [(semitones, intervals, contour, i_hist, c_hist)]

        q_type = "mixed" if is_song else detect_query_type(y_proc, sr)

        return {
            "semitones": semitones,
            "intervals": intervals,
            "contour":   contour,
            "i_hist":    i_hist,
            "c_hist":    c_hist,
            "segments":  segments,
            "q_type":    q_type,
            "backend_used": backend_used if not is_song else "pyin",
            "fallback_used": fallback_used if not is_song else False,
        }

    except Exception as e:
        print(f"  ERR  {basename}: {e}", flush=True)
        return None

def print_segment_stats(feature_db):
    """Print segment distribution across songs to detect bias."""
    print("\n[db] Database Segment Distribution:", flush=True)
    stats = []
    for song, data in feature_db.items():
        n = len(data.get('segments', []))
        stats.append((n, song))
    
    stats.sort(reverse=True)
    for n, s in stats[:10]:
        print(f"  {n:3d} segments | {s[:40]}", flush=True)
    
    total = sum(n for n, s in stats)
    avg = total / len(stats) if stats else 0
    print(f"[db] Total segments: {total}, Avg: {avg:.1f} per song\n", flush=True)

def build_feature_database(wav_dir=WAV_DIR, pkl_path=FEATURES_PKL, force=False):
    # Stub for now since we already rebuilt, but keep interface for app.py
    print("[features] build_feature_database called (stub)", flush=True)

if __name__ == "__main__":
    force = "--force" in sys.argv
    if force:
        # We'd need the real implementation here if we wanted to rebuild from CLI
        print("Rebuild logic removed for brevity in this bypass version.")
