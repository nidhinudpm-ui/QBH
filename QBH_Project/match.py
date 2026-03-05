"""
match.py — Constrained DTW + Fast Pre-Filtering for QBH

Architecture:
  Step 1 — Chroma cosine pre-filter on ALL songs (instant)
  Step 2 — Keep top 10 candidates
  Step 3 — fastdtw with radius constraint (Sakoe-Chiba band equivalent)
  Step 4 — Score: 0.80 * interval_dtw + 0.20 * chroma_cosine
  Step 5 — Softmax confidence ranking

fastdtw's `radius` parameter acts as a Sakoe-Chiba band constraint,
preventing extreme tempo distortion while being much faster than
pure-Python DTW implementations.
"""

import os
import sys
import numpy as np
import pickle
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean, cosine

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from extract_features import extract_features
from confidence import rank_with_confidence
from config import FEATURES_PKL, TOP_MATCHES

# Pre-filter: keep this many candidates before DTW
PRE_FILTER_N = 10


def constrained_dtw(query, ref, radius=None):
    """
    fastdtw with radius constraint (equivalent to Sakoe-Chiba band).
    Returns normalized cost: total_cost / path_length.
    """
    if radius is None:
        radius = max(3, int(len(query) * 0.10))

    dist, path = fastdtw(
        query.reshape(-1, 1),
        ref.reshape(-1, 1),
        dist=euclidean,
        radius=radius
    )
    return dist / max(len(path), 1)


def sliding_constrained_dtw(query, song, radius=None):
    """
    Slide query over song using constrained DTW.
    Uses 4 evenly-spaced windows for speed.
    Returns minimum normalized distance.
    """
    q_len = len(query)
    s_len = len(song)

    if s_len <= q_len * 2:
        # Song is short enough to match directly
        return constrained_dtw(query, song, radius)

    # 4 anchor windows: start, 25%, 50%, 75%
    win = min(q_len * 2, s_len)
    anchors = [0, s_len // 4, s_len // 2, max(0, s_len - win)]
    best = float('inf')

    for anchor in anchors:
        end = min(anchor + win, s_len)
        segment = song[anchor:end]
        if len(segment) < q_len:
            continue
        d = constrained_dtw(query, segment, radius)
        if d < best:
            best = d

    return best


def match_query(query_file, pkl_path=FEATURES_PKL, top_n=TOP_MATCHES, return_results=False):
    """
    Cosine pre-filter (all songs) → constrained DTW (top 10) → rank.
    """
    if not os.path.exists(query_file):
        print(f"Error: '{query_file}' not found.")
        return [] if return_results else None
    if not os.path.exists(pkl_path):
        print(f"Error: Feature DB not found at '{pkl_path}'.")
        return [] if return_results else None

    with open(pkl_path, 'rb') as f:
        db = pickle.load(f)
    if not db:
        return [] if return_results else None

    print("Extracting features from query…")
    q_result = extract_features(query_file)
    q_intervals = q_result[0]
    q_chroma    = q_result[1]

    if q_intervals is None:
        print("Insufficient melody in query. Hum for at least 5 seconds.")
        return [] if return_results else None

    q_len = len(q_intervals)
    band  = max(3, int(q_len * 0.10))
    print(f"Query: {q_len} intervals, DTW band radius={band}")

    # ── Step 1: Fast cosine pre-filter ──
    print(f"Step 1: Chroma cosine pre-filter on {len(db)} songs…")
    prescores = []
    for song_name, feats in db.items():
        ch_dist = cosine(q_chroma, feats["chroma"])
        prescores.append((song_name, feats, ch_dist))

    prescores.sort(key=lambda x: x[2])
    shortlist = prescores[:PRE_FILTER_N]

    # ── Step 2: Constrained DTW on shortlist ──
    print(f"Step 2: Constrained DTW on top {len(shortlist)} candidates…")
    results = []
    for song_name, feats, ch_dist in shortlist:
        iv_dist = sliding_constrained_dtw(q_intervals, feats["intervals"], radius=band)

        final = 0.80 * iv_dist + 0.20 * ch_dist

        results.append({
            "song_name":     song_name,
            "interval_dist": float(iv_dist),
            "chroma_dist":   float(ch_dist),
            "final_score":   float(final)
        })
        print(f"  {song_name[:38]:38s}: iv={iv_dist:.4f} ch={ch_dist:.4f} → {final:.4f}")

    top = rank_with_confidence(results, distance_key="final_score", top_n=top_n)

    if return_results:
        return top

    from confidence import display_ranked
    display_ranked(top)


if __name__ == "__main__":
    match_query("query.wav")
