import numpy as np
import time
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean, cosine
from scipy.stats import pearsonr
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from config import (
    W_INTERVAL_DDTW, W_CONTOUR_DTW, W_INTERVAL_HIST, W_RHYTHM,
    HIST_REJECT_THRESHOLD
)
from melody_features import compute_interval_histogram, compute_contour_histogram

# ─── Optimization Constants ──────────────────────────────────────────────────
FAST_DTW_RADIUS = 5
MAX_SEGMENTS_PER_SONG = 5  # DTW cap per song after coarse-ranking
# Note: HIST_REJECT_THRESHOLD is imported from config

# ─── Utility ─────────────────────────────────────────────────────────────────

def downsample(seq, factor=2):
    """Simple 1D downsampling."""
    if seq is None or len(seq) < factor * 2:
        return seq
    return seq[::factor]

# ─── Core DTW Functions ──────────────────────────────────────────────────────

def compute_derivative(seq):
    """Compute first derivative of a sequence for DDTW."""
    if len(seq) < 3:
        return seq.copy()
    deriv = np.zeros_like(seq)
    deriv[0] = seq[1] - seq[0]
    deriv[-1] = seq[-1] - seq[-2]
    for i in range(1, len(seq) - 1):
        deriv[i] = (seq[i + 1] - seq[i - 1]) / 2.0
    return deriv

def weighted_dist(p1, p2):
    """Custom distance: weight * abs(val1 - val2). p1[1] is the weight."""
    return p1[1] * abs(p1[0] - p2[0])

def compute_movement_weights(seq, threshold=0.2): # Relaxed threshold to be less sensitive
    """Assign weights: 1.0 for moving frames, 0.2 for flat ones."""
    if len(seq) == 0: return np.array([])
    # Use absolute difference from neighbors
    diffs = np.zeros_like(seq)
    if len(seq) > 1:
        diffs[0] = abs(seq[1] - seq[0])
        diffs[-1] = abs(seq[-1] - seq[-2])
        for i in range(1, len(seq) - 1):
            diffs[i] = (abs(seq[i+1] - seq[i]) + abs(seq[i] - seq[i-1])) / 2.0
    
    # User requested: moving=1.0, flat=0.2
    weights = np.where(diffs >= threshold, 1.0, 0.2)
    return weights

def subsequence_dtw(query, target, use_weights=False):
    """Subsequence DTW with optional movement weighting."""
    if len(query) == 0 or len(target) == 0:
        return float('inf'), [], False

    # 2x Downsampling
    q_ds = downsample(query, 2)
    t_ds = downsample(target, 2)
    
    q_len = len(q_ds)
    t_len = len(t_ds)

    # ─── Symmetric Matching Logic ───
    if q_len > t_len:
        fixed_seq, sliding_seq = t_ds, q_ds
        is_swapped = True
    else:
        fixed_seq, sliding_seq = q_ds, t_ds
        is_swapped = False

    f_len = len(fixed_seq)
    s_len = len(sliding_seq)

    # Prepare weights if requested
    if use_weights:
        f_weights = compute_movement_weights(fixed_seq)
        # Pack as [[val, weight], ...]
        fixed_packed = np.column_stack((fixed_seq, f_weights))
        sliding_packed = np.column_stack((sliding_seq, np.ones(s_len))) # Sliding weights neutral
        dist_fn = weighted_dist
    else:
        fixed_packed = fixed_seq.reshape(-1, 1)
        sliding_packed = sliding_seq.reshape(-1, 1)
        dist_fn = euclidean

    # Bug #2 Fix: Scale DTW radius with sequence length.
    # A fixed radius of 5 is too tight for real humming tempo variation.
    dynamic_radius = max(10, f_len // 5)

    # If sequences are similar in length, just do direct DTW
    if s_len <= f_len * 1.5:
        dist, path = fastdtw(fixed_packed, sliding_packed, dist=dist_fn, radius=dynamic_radius)
        norm = dist / max(len(path), 1)
        if is_swapped:
            path = [(j, i) for i, j in path]
        return norm, path, is_swapped

    # Otherwise, sliding window over the longer sequence
    win_size = min(int(f_len * 1.5), s_len)
    step = max(1, f_len // 3)
    best_dist = float('inf')
    best_path = []

    for start in range(0, s_len - f_len + 1, step):
        end = min(start + win_size, s_len)
        window_packed = sliding_packed[start:end]

        dist, path = fastdtw(fixed_packed, window_packed, dist=dist_fn, radius=dynamic_radius)
        norm = dist / max(len(path), 1)
        if norm < best_dist:
            best_dist = norm
            if is_swapped:
                best_path = [(j + start, i) for i, j in path]
            else:
                best_path = [(i, j + start) for i, j in path]

    return best_dist, best_path, is_swapped

def subsequence_ddtw(query, target):
    """Subsequence DDTW (Restored movement weighting for accuracy)."""
    q_deriv = compute_derivative(query)
    t_deriv = compute_derivative(target)
    return subsequence_dtw(q_deriv, t_deriv, use_weights=True) # Accurate weighting

def compute_shape_correlation(q_vals, s_vals):
    """Pearson correlation on matched sequences, with user-requested guards."""
    # Guard 1: Path length (User requested > 50)
    if len(q_vals) < 50 or len(s_vals) < 50:
        return 0.0
    
    # Guard 2: Matched-subsequence variance (User requested > 0.1)
    if np.std(q_vals) < 0.1 or np.std(s_vals) < 0.1:
        return 0.0
        
    try:
        corr, _ = pearsonr(q_vals, s_vals)
        # Monotonicity: only positive correlation is a 'bonus'
        return float(max(0.0, corr)) if not np.isnan(corr) else 0.0
    except:
        return 0.0

def compute_landmark_agreement(q_vals, s_vals):
    """Lightweight agreement: do both rise/fall in the same places?"""
    if len(q_vals) < 10: return 0.0
    
    q_slopes = np.diff(q_vals)
    s_slopes = np.diff(s_vals)
    
    # Compare signs of slopes (-1 fallback for 0/flat)
    q_signs = np.sign(q_slopes)
    s_signs = np.sign(s_slopes)
    
    # Filter out perfectly flat frames from both for agreement check
    moving_mask = (q_signs != 0) | (s_signs != 0)
    if not np.any(moving_mask): return 0.0
    
    agreement = np.mean(q_signs[moving_mask] == s_signs[moving_mask])
    return float(agreement)

def extract_matched_sequences(path, q_intervals, s_intervals, q_semi, s_semi, ds_factor=2):
    """
    Extract query and song subsequences based on the DTW path.
    'path' is in downsampled space.
    """
    if not path:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    q_idxs = [i for i, j in path]
    s_idxs = [j for i, j in path]
    
    mq_int = q_intervals[[min(len(q_intervals)-1, i * ds_factor) for i in q_idxs]]
    ms_int = s_intervals[[min(len(s_intervals)-1, j * ds_factor) for j in s_idxs]]
    mq_semi = q_semi[[min(len(q_semi)-1, i * ds_factor) for i in q_idxs]]
    ms_semi = s_semi[[min(len(s_semi)-1, j * ds_factor) for j in s_idxs]]
    
    return mq_int, ms_int, mq_semi, ms_semi

def compute_movement_fraction(seq, threshold=0.5):
    """Fraction of frames with significant melodic movement."""
    if len(seq) == 0: return 0.0
    return float(np.mean(np.abs(seq) >= threshold))

# ─── Segment-Level Scoring ───────────────────────────────────────────────────

def score_segment(query_features, segment_features, skip_coarse=False):
    """
    Score a query against a single song segment with high-discrimination penalties.
    """
    from config import DDTW_EARLY_EXIT_THRESH
    
    # Unpack features
    q_semi, q_intervals, q_contour, q_i_hist, q_c_hist = query_features
    
    if len(segment_features) == 5:
        s_semi, s_intervals, s_contour, s_i_hist, s_c_hist = segment_features
    else:
        s_semi, s_intervals, s_contour = segment_features
        s_i_hist = compute_interval_histogram(s_intervals)
        s_c_hist = None

    if len(q_intervals) < 3 or len(s_intervals) < 3:
        return 0.0, None

    # 1. Improved Coarse Ranking (Combined Histogram)
    d_i_hist = float(cosine(q_i_hist, s_i_hist)) if np.any(q_i_hist) and np.any(s_i_hist) else 1.0
    d_c_hist = 1.0
    if q_c_hist is not None and s_c_hist is not None:
        if np.any(q_c_hist) and np.any(s_c_hist):
            d_c_hist = float(cosine(q_c_hist, s_c_hist))
    
    # Relaxed for Phase 11
    d_hist = 0.6 * d_i_hist + 0.4 * d_c_hist
    if not skip_coarse and d_hist > (HIST_REJECT_THRESHOLD + 0.15):
        return 0.0, None

    # 2. Length Ratio Basic Sanity (Original Space)
    ratio = len(s_intervals) / max(len(q_intervals), 1)
    from config import LEN_RATIO_MIN
    # Phase 12: Standardized dynamic max ratio
    max_ratio = 4.5 if len(q_intervals) > 60 else 7.0
    if not skip_coarse and (ratio < LEN_RATIO_MIN or ratio > max_ratio):
        return 0.0, None

    # 3. Interval DDTW Matching
    d_ddtw, path, is_swapped = subsequence_ddtw(q_intervals, s_intervals)
    path_len = len(path) if path else 0

    # 4. Contour DTW Matching
    d_contour, c_path, _ = subsequence_dtw(q_contour, s_contour)

    # 5. Weighted composite distance
    d_rhythm = 0.0
    total_dist = (
        W_INTERVAL_DDTW * d_ddtw +
        W_CONTOUR_DTW   * d_contour +
        W_INTERVAL_HIST * d_hist +
        W_RHYTHM        * d_rhythm
    )

    # Ranking Score: Gentle mapping (1 / (1 + d))
    melody_score = 1.0 / (1.0 + total_dist)

    # ─── Length Constraints (Downsampled Frame Space) ───
    # path_len is already in downsampled space
    q_len_ds = max(1, len(q_intervals) // 2)
    len_ratio = len(path) / q_len_ds if q_len_ds > 0 else 0
    
    # Coverage relative to query (Phase 10: relaxed for 2s clips)
    # We want to ensure at least 50% of the query matched SOMETHING in the song
    if len_ratio < 0.4: # Query coverage
        return 0.0, None

    # Coverage relative to song segment (Subsequence check)
    # 41 frames (2s) vs 375 frames (6s) is ~0.11. 
    # Use 0.08 as hard limit for very brief distinctive motifs.
    s_len_ds = max(1, len(s_intervals) // 2)
    s_coverage = len(path) / s_len_ds if s_len_ds > 0 else 0
    if s_coverage < 0.08:
        return 0.0, None

    # Bug #3 Fix: Old formula was a dead no-op (produced 1.0 for all legal s_coverage values).
    # New formula: 0.6 at minimum coverage (0.08), smoothly rising to 1.0 at 50%+ coverage.
    coverage_penalty = min(1.0, 0.6 + 0.8 * s_coverage)
    melody_score *= coverage_penalty

    # 7. Warp Penalty
    s_len_ds = max(1, len(s_intervals) // 2)
    max_len = max(q_len_ds, s_len_ds)
    warp_ratio = len(path) / max_len if max_len > 0 else 1.0
    warp_penalty = 1.0
    if warp_ratio > 1.15:
        warp_penalty = max(0.2, 1.0 - (warp_ratio - 1.15) * 2.0)
        melody_score *= warp_penalty

    # ─── New Matched-Path Quality Analysis ───
    if path:
        # Extract matched subsequences in original (non-downsampled) space
        mq_int, ms_int, mq_semi, ms_semi = extract_matched_sequences(
            path, q_intervals, s_intervals, q_semi, s_semi, ds_factor=2
        )
        
        # 1. Basic Density
        non_flat_s = ms_int[np.abs(ms_int) >= 1.0]
        unique_intervals = len(np.unique(non_flat_s))
        zero_fraction = np.mean(np.abs(ms_int) < 1.0)
        interval_std = np.std(non_flat_s) if len(non_flat_s) > 1 else 0.0
        
        # 2. Movement Agreement
        q_move = compute_movement_fraction(mq_int, 0.5)
        s_move = compute_movement_fraction(ms_int, 0.5)
        movement_agreement = 1.0 - abs(q_move - s_move)
        
        # 3. Shape Agreement (Pearon on intervals + semitones)
        shape_corr_int = compute_shape_correlation(mq_int, ms_int)
        shape_corr_semi = compute_shape_correlation(mq_semi, ms_semi)
        shape_correlation = max(shape_corr_int, shape_corr_semi)
        
        # 4. Landmark Agreement (Rise/Fall)
        landmark_int = compute_landmark_agreement(mq_int, ms_int)
        landmark_semi = compute_landmark_agreement(mq_semi, ms_semi)
        landmark_score = max(landmark_int, landmark_semi)
        
        # 5. Combined Shape Quality
        shape_quality = 0.6 * shape_correlation + 0.4 * landmark_score
    else:
        unique_intervals = 0
        zero_fraction = 1.0
        interval_std = 0.0
        movement_agreement = 0.0
        shape_correlation = 0.0
        landmark_score = 0.0
        shape_quality = 0.0
        mq_semi, ms_semi = np.array([]), np.array([])
    
    # ─── Tiered Matched-Path Info Penalty (Restored for accuracy) ───
    info_penalty = 1.0
    # Strong penalty for zero_fraction > 0.65
    if zero_fraction > 0.65:
        info_penalty *= np.exp(-(zero_fraction - 0.65) * 3.0)
    
    # Moderate penalty for unique_intervals < 4 (Phase 10: relaxed for short queries)
    min_unique = 4 if len(q_intervals) > 60 else 2
    if unique_intervals < min_unique:
        info_penalty *= max(0.45, unique_intervals / min_unique)
        
    # Mild penalty for low interval_std
    if interval_std < 0.35: # Changed from 0.3
        info_penalty *= 0.80 # Made stronger for flat matches
    
    melody_score *= info_penalty

    # ─── Shape-Agreement Refinement (Restored guards) ───
    shape_multiplier = 1.0
    if path_len > 65 and interval_std > 0.15: # Stronger path-length guard for shape bonus
        # Boost up to 15%, Penalize down to 15% (0.85 to 1.15)
        if shape_quality > 0.5:
            # Positive boost
            shape_multiplier = 1.0 + (shape_quality - 0.5) * 0.30  # Max 1.15
        else:
            # Natural penalty
            shape_multiplier = 0.85 + 0.15 * (shape_quality / 0.5)
    melody_score *= shape_multiplier

    # ─── Movement Refinement ───
    # Mild multiplier based on movement agreement
    melody_score *= (0.9 + 0.1 * movement_agreement)

    # ─── Tiny Richness Bonus ───
    richness_bonus = 1.0
    if unique_intervals > 8 and interval_std > 2.0:
        richness_bonus = min(1.10, 1.0 + (unique_intervals - 8) * 0.01)
    melody_score *= richness_bonus

    # ─── Waveform Readiness (Always ON for top candidates) ───
    waveform_ok = True

    # Locally normalized display contours
    aligned_q_display = []
    aligned_s_display = []
    if waveform_ok and len(mq_semi) > 2:
        q_mean = np.mean(mq_semi)
        s_mean = np.mean(ms_semi)
        q_std = np.std(mq_semi)
        s_std = np.std(ms_semi)
        
        # Avoid div by zero
        q_std = max(q_std, 0.5)
        s_std = max(s_std, 0.5)
        
        # Display normalization: Simple z-score (frontend handles range)
        norm_q = (mq_semi - q_mean) / q_std
        norm_s = (ms_semi - s_mean) / s_std
        
        aligned_q_display = [float(x) for x in norm_q[:600]]
        aligned_s_display = [float(x) for x in norm_s[:600]]

    # Build exhaustive match info
    match_info = {
        "d_ddtw":    float(d_ddtw),
        "d_contour": float(d_contour),
        "d_hist":    float(d_hist),
        "total_dist": float(total_dist),
        "path_len":  path_len,
        "len_ratio": float(len_ratio),
        "warp_ratio": float(warp_ratio),
        "warp_penalty": float(warp_penalty),
        "zero_fraction": float(zero_fraction),
        "unique_intervals": int(unique_intervals),
        "interval_std": float(interval_std),
        "info_penalty": float(info_penalty),
        "movement_agreement": float(movement_agreement),
        "shape_correlation": float(shape_correlation),
        "landmark_score": float(landmark_score),
        "shape_multiplier": float(shape_multiplier),
        "richness_bonus": float(richness_bonus),
        "waveform_ok": waveform_ok,
        "aligned_q_display": aligned_q_display,
        "aligned_s_display": aligned_s_display,
        "early_exit": False,
        "s_coverage": float(s_coverage),
    }

    return float(melody_score), match_info

# ─── Song-Level Matching ────────────────────────────────────────────────────

def match_query_to_song(query_features, song_data, song_name=""):
    """
    Find best segment using coarse-ranking and DTW cap.
    """
    segments = song_data.get("segments", [])
    if not segments:
        return 0.0, None

    q_i_hist = query_features[3]
    
    # --- Coarse Ranking Stage ---
    candidates = []
    for i, seg in enumerate(segments):
        # Extract histograms (prefer precomputed)
        if len(seg) >= 5:
            s_i_hist = seg[3]
            s_c_hist = seg[4]
        else:
            s_i_hist = compute_interval_histogram(seg[1])
            s_c_hist = None
            
        d_i_hist = cosine(q_i_hist, s_i_hist) if np.any(q_i_hist) and np.any(s_i_hist) else 1.0
        d_c_hist = 1.0
        if s_c_hist is not None and query_features[4] is not None:
            if np.any(query_features[4]) and np.any(s_c_hist):
                d_c_hist = cosine(query_features[4], s_c_hist)
        
        d_hist = 0.6 * d_i_hist + 0.4 * d_c_hist
        
        # ─── Combined Coarse Rejection ───
        # Bug #6 Fix: Threshold of 1.1 exceeded cosine distance max (1.0), so nothing
        # was ever rejected for short queries — making the filter a complete no-op.
        # Now use a moderately relaxed threshold instead of disabling it entirely.
        reject_thresh = HIST_REJECT_THRESHOLD + 0.25
        if len(query_features[1]) < 60:  # Short query: relax but don't disable
            reject_thresh = HIST_REJECT_THRESHOLD + 0.30
            
        if d_hist > reject_thresh: 
            continue
            
        # Reject by length sanity (original space)
        ratio = len(seg[1]) / max(len(query_features[1]), 1)
        
        # User requested tightened limits: 0.18 min, and 4.5/7.0 for varying lengths
        min_ratio = 0.18
        max_ratio = 4.5 if len(query_features[1]) > 60 else 7.0
        if ratio < min_ratio or ratio > max_ratio:
            continue
            
        candidates.append((d_hist, i, seg))

    if not candidates:
        return 0.0, None

    # Sort by histogram distance (best first)
    candidates.sort(key=lambda x: x[0])
    
    # User requested: rank segments using combined histogram score, 
    # then run DTW on top 3 segments to tighten false positives.
    to_eval = candidates[:3]

    best_score = 0.0
    best_info = None

    for d_hist, idx, seg in to_eval:
        score, info = score_segment(query_features, seg, skip_coarse=True)
        

        if score > best_score:
            best_score = score
            best_info = info
            if best_info:
                best_info["segment_index"] = idx

    return best_score, best_info

def rank_songs_by_melody(query_features, feature_db, top_n=10):
    """
    Rank all songs with optimized pipeline and timing logs.
    """
    start_time = time.time()
    rankings = []
    dtw_segments_count = 0

    print(f"\n[melody] Starting matching for {len(feature_db)} songs...", flush=True)

    # Add histograms to query_features if missing
    if len(query_features) < 4:
        # Unexpected feature tuple format
        pass
    elif len(query_features) == 4:
        # Just missing contour histogram (legacy or simple query)
        q_c_hist = compute_contour_histogram(query_features[2])
        query_features = (*query_features, q_c_hist)
    # else: query_features already has 5 items (packed correctly)

    for song_name, song_data in feature_db.items():
        if "thumbi" in song_name.lower():
             print(f"  [debug] Found target in loop: {song_name}", flush=True)
             
        song_start = time.time()
        
        # Segment Count Inspection (Phase 6)
        n_seg = len(song_data.get("segments", []))
        if n_seg > 150: # Unusually high
            print(f"  [warn] {song_name[:30]} has {n_seg} segments (Bias Risk)", flush=True)
            
        score, info = match_query_to_song(query_features, song_data, song_name=song_name)
        
        if score > 0 and info is not None:
            rankings.append({
                "song_name":    song_name,
                "melody_score": score,
                "match_info":   info,
            })
            # Real DTW count from top_n segments evaluated per song
            dtw_segments_count += min(n_seg, MAX_SEGMENTS_PER_SONG)
            
        elapsed = time.time() - song_start
        # Verbose log for long-running songs if any
        if elapsed > 0.5:
            print(f"  [warn] {song_name[:30]} took {elapsed:.2f}s", flush=True)

    rankings.sort(key=lambda x: x["melody_score"], reverse=True)
    
    total_time = time.time() - start_time
    print(f"[melody] Finished. Total songs: {len(feature_db)}, Matches: {len(rankings)}", flush=True)
    print(f"[melody] Overall time: {total_time:.2f}s", flush=True)
    
    return rankings[:top_n]
