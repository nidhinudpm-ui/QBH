"""
extract_features.py — F0-Centric Feature Extraction for QBH

Pipeline:
  1. Vocal isolation (center channel for songs)
  2. Preprocessing (audio_preprocess module)
  3. pYIN F0 extraction → continuous semitone contour
  4. Fallback to Basic Pitch if pYIN fails
  5. Interval + contour + histogram features
  6. Segmentation (songs only, 6s windows with 50% overlap)
  7. Chroma mean (for recommendation engine)
  8. Query-type detection (queries only)

Feature schema (no legacy keys):
  semitones, intervals, contour,
  i_hist, c_hist, chroma,
  segments [(semitones, intervals, contour), ...],
  q_type
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
    SAMPLE_RATE, CONFIDENCE_THRESHOLD, MIN_NOTE_DURATION,
    SEGMENT_LENGTH_SEC, SEGMENT_OVERLAP, PYIN_FMIN, PYIN_FMAX,
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

# WAV_DIR and FEATURES_PKL imported from config


# ═══════════════════════════════════════════════════════════════════════════════
#  VOCAL ISOLATION (songs only — center channel extraction)
# ═══════════════════════════════════════════════════════════════════════════════

def get_vocal_path(file_path):
    """
    Center-channel extraction for stereo tracks.
    Vocals are usually panned center (high L/R correlation).
    Lightweight alternative to Demucs/Spleeter.
    """
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=False)

        if y.ndim == 1:
            return file_path  # Already mono

        # Mid = (L + R) / 2
        mid = (y[0] + y[1]) / 2.0

        basename = os.path.basename(file_path)
        temp_path = os.path.join(PROJECT_DIR, "uploads", f"vocal_{basename}")
        import soundfile as sf
        sf.write(temp_path, mid, SAMPLE_RATE)

        return temp_path
    except Exception as e:
        print(f"    [vocal] Center-channel failed: {e}. Using original.")
        return file_path


# ═══════════════════════════════════════════════════════════════════════════════
#  BASIC PITCH FALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

def run_basic_pitch(file_path):
    """
    Run Basic Pitch on an audio file.
    Returns list of (start_time, end_time, midi_pitch, confidence, duration).
    """
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH

    model_output, midi_data, note_events = predict(
        file_path,
        ICASSP_2022_MODEL_PATH
    )

    notes = []
    for note in note_events:
        start_time = float(note[0])
        end_time   = float(note[1])
        pitch      = int(note[2])
        confidence = float(note[3]) / 127.0 if note[3] > 1 else float(note[3])
        duration   = end_time - start_time
        notes.append((start_time, end_time, pitch, confidence, duration))

    return notes


def notes_to_pseudo_contour(notes):
    """Convert Basic Pitch note events into a pseudo-semitone contour."""
    if not notes:
        return None
    dur = notes[-1][1]
    frames = int((dur * SAMPLE_RATE) / F0_HOP_LENGTH)
    if frames < 5:
        return None
    semitones = np.zeros(frames, dtype=np.float32)
    for s, e, p, c, d in notes:
        s_f = int((s * SAMPLE_RATE) / F0_HOP_LENGTH)
        e_f = int((e * SAMPLE_RATE) / F0_HOP_LENGTH)
        semitones[s_f:min(e_f, frames)] = p
    return semitones


# ═══════════════════════════════════════════════════════════════════════════════
#  F0 CONTOUR EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_f0_contour(file_path, is_query=True):
    """
    Extract continuous F0 semitone contour.
    Query: preprocess with full pipeline.
    Song: use vocal isolation first.
    """
    try:
        if is_query:
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
            
            # --- Aggressive Voiced-Region Trimming (Phase 9) ---
            # Trim leading/trailing silence to reduce pYIN workload
            yt, index = librosa.effects.trim(y, top_db=30)
            if len(yt) > (sr * 0.5): # Only use trimmed if at least 0.5s remains
                y = yt
                
            y = preprocess_query_audio(y, sr)
        else:
            vocal_path = get_vocal_path(file_path)
            y, sr = librosa.load(vocal_path, sr=SAMPLE_RATE, mono=True)
            y = preprocess_song_audio(y, sr)

        semitones = get_continuous_contour(
            y, sr,
            fmin=PYIN_FMIN, fmax=PYIN_FMAX,
            hop_length=F0_HOP_LENGTH
        )
        return semitones
    except Exception as e:
        print(f"    [f0] Extraction failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def segment_f0_contour(semitones, segment_sec=SEGMENT_LENGTH_SEC, overlap=SEGMENT_OVERLAP):
    """Segment continuous F0 contour into fixed-length windows."""
    seg_frames = int((segment_sec * SAMPLE_RATE) / F0_HOP_LENGTH)
    step_frames = int(seg_frames * (1 - overlap))
    n_frames = len(semitones)

    if n_frames < seg_frames:
        return [semitones]

    segments = []
    for start in range(0, n_frames - seg_frames + 1, step_frames):
        seg = semitones[start:start + seg_frames]
        # Keep segment if at least 15% voiced
        if np.mean(seg > 0) > 0.15:
            segments.append(seg)

    return segments if segments else [semitones]


# ═══════════════════════════════════════════════════════════════════════════════
#  CHROMA (for recommendation engine only)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_chroma_mean(file_path, sr=SAMPLE_RATE):
    """Compute mean chroma vector for recommendation engine compatibility."""
    try:
        y, sr = librosa.load(file_path, sr=sr, mono=True)
        y = librosa.util.normalize(y)
        y_harmonic, _ = librosa.effects.hpss(y)

        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=512)
        norms = np.linalg.norm(chroma, axis=0, keepdims=True)
        norms[norms == 0] = 1
        chroma_mean = np.mean(chroma / norms, axis=1)

        cn = np.linalg.norm(chroma_mean)
        if cn > 0:
            chroma_mean = chroma_mean / cn

        return chroma_mean.astype(np.float32)
    except Exception:
        return np.zeros(12, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features(file_path, is_song=False):
    """
    F0-Centric Feature Extraction Pipeline.
    """
    try:
        basename = os.path.basename(file_path)

        # ── Step 1: Continuous F0 ──
        print(f"    [f0] Tracking melody in {basename[:45]}...", flush=True)
        
        # Load audio once to reuse for type detection and F0
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        
        if not is_song:
            # Crop query to MAX_RECORDING_SEC for performance
            max_samples = MAX_RECORDING_SEC * sr
            if len(y) > max_samples:
                print(f"    [extract] Cropping query {len(y)/sr:.1f}s -> {MAX_RECORDING_SEC}s", flush=True)
                y = y[:max_samples]

            # Aggressive silence trimming
            y, _ = librosa.effects.trim(y, top_db=25)
            y_proc = preprocess_query_audio(y, sr)

            # Pitch Extraction
            result = get_query_contour(y_proc, sr, backend=QUERY_PITCH_BACKEND)
            semitones = result["contour"]
            backend_used = result["backend_used"]
            fallback_used = result["fallback_used"]
            
            # Trim to active voiced region
            voiced_idxs = np.where(semitones > 0)[0]
            if len(voiced_idxs) > 0:
                semitones = semitones[voiced_idxs[0]:voiced_idxs[-1]+1]
            
            # Cap for DTW safety
            MAX_QUERY_FRAMES = 1200
            if len(semitones) > MAX_QUERY_FRAMES:
                semitones = semitones[:MAX_QUERY_FRAMES]
        else:
            # Vocal isolation for songs
            vocal_path = get_vocal_path(file_path)
            y_vocal, sr = librosa.load(vocal_path, sr=SAMPLE_RATE, mono=True)
            y_proc = preprocess_song_audio(y_vocal, sr)

            # Shared pYIN path
            semitones = get_continuous_contour(
                y_proc, sr,
                fmin=PYIN_FMIN, fmax=PYIN_FMAX,
                hop_length=F0_HOP_LENGTH
            )

        # Fallback Logic (Song path)
        if (semitones is None or np.mean(semitones > 0) < 0.05) and is_song:
            print("    [fallback] Sparse melody. Using Basic Pitch proxy.", flush=True)
            notes = run_basic_pitch(file_path)
            semitones = notes_to_pseudo_contour(notes)
        
        if semitones is None or len(semitones) < 5:
            return None

        # ── Step 3: Feature Generation ──
        intervals = compute_intervals(semitones)
        contour = compute_contour(intervals)
        i_hist = compute_interval_histogram(intervals)
        c_hist = compute_contour_histogram(contour)

        # ── Step 4: Segmentation (songs only) ──
        if is_song:
            raw_segments = segment_f0_contour(semitones)
            segments = []
            for seg in raw_segments:
                seg_intervals = compute_intervals(seg)
                seg_contour = compute_contour(seg_intervals)
                seg_i_hist = compute_interval_histogram(seg_intervals)
                seg_c_hist = compute_contour_histogram(seg_contour)
                # Schema: (semitones, intervals, contour, i_hist, c_hist)
                segments.append((seg, seg_intervals, seg_contour, seg_i_hist, seg_c_hist))
        else:
            segments = [(semitones, intervals, contour, i_hist, c_hist)]

        # ── Step 5: Type Detection ──
        q_type = "mixed"
        if not is_song:
            q_type = detect_query_type(y_proc, sr)

        return {
            "semitones": semitones,
            "intervals": intervals,
            "contour":   contour,
            "i_hist":    i_hist,
            "c_hist":    c_hist,
            "segments":  segments,
            "q_type":    q_type,
        }

    except Exception as e:
        print(f"  ERR  {basename}: {e}", flush=True)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  DATABASE BUILD
# ═══════════════════════════════════════════════════════════════════════════════

def _process_single_song(args):
    """Worker function for parallel extraction."""
    filepath, filename = args
    try:
        result = extract_features(filepath, is_song=True)
        if result is not None:
            return (filename, result)
        return None
    except Exception as e:
        print(f"  ERR  {filename}: {e}")
        return None


def _parallel_worker(args):
    """Top-level worker function for parallel extraction."""
    fp, fn = args
    try:
        # Re-import inside worker to ensure clean state in spawned processes
        from extract_features import extract_features
        res = extract_features(fp, is_song=True)
        return fn, res
    except Exception as e:
        print(f"  [worker] Error on {fn}: {e}")
        return fn, None


def build_feature_database(wav_dir=WAV_DIR, pkl_path=FEATURES_PKL, force=False):
    """
    Extract features for all WAV files and save to saved_features.pkl.
    Uses parallel processing for speed.
    """
    if os.path.exists(pkl_path) and not force:
        print(f"[features] Database already exists -> '{pkl_path}'. Skipping.")
        print(f"[features] Run with --force to rebuild.")
        return

    if not os.path.exists(wav_dir):
        print(f"Error: '{wav_dir}' not found. Run convert.py first.")
        return

    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)

    # Allow both .wav and .mp3
    extensions = ('.wav', '.mp3')
    files = sorted(f for f in os.listdir(wav_dir) if f.lower().endswith(extensions))
    if not files:
        print(f"No WAV files in '{wav_dir}'. Run convert.py first.")
        return

    print(f"\n{'='*60}")
    print(f"  Building F0-Centric Feature DB (Parallel)")
    print(f"  Songs: {len(files)}")
    print(f"{'='*60}\n")

    import concurrent.futures
    db = {}
    
    tasks = [(os.path.join(wav_dir, fn), fn) for fn in files]

    # Use max_workers=4 to avoid OOM or CPU saturation
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        future_to_song = {executor.submit(_parallel_worker, t): t[1] for t in tasks}
        
        count = 0
        for future in concurrent.futures.as_completed(future_to_song):
            count += 1
            fn = future_to_song[future]
            try:
                res = future.result()
                if res and res[1]:
                    name, data = res
                    db[name] = data
                    n_seg = len(data.get("segments", []))
                    print(f"[{count}/{len(files)}] {fn[:50]:50s} OK ({n_seg} segments)")
                else:
                    print(f"[{count}/{len(files)}] {fn[:50]:50s} SKIP")
            except Exception as e:
                print(f"[{count}/{len(files)}] {fn[:50]:50s} ERROR: {e}")

    if not db:
        print("\n[!] FATAL: No features extracted. Check errors above.")
        return

    with open(pkl_path, 'wb') as f:
        pickle.dump(db, f)

    print(f"\n{'='*60}")
    print(f"  Done! {len(db)}/{len(files)} songs saved to")
    print(f"     '{pkl_path}'")
    print(f"{'='*60}\n")


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

if __name__ == "__main__":
    force = "--force" in sys.argv
    build_feature_database(force=force)
