"""
confidence.py — Softmax-based confidence scoring

Converts raw DTW/hybrid distances into user-friendly percentage scores
using softmax over negative distances, ensuring top matches sum to ~100%.
"""

import numpy as np


def softmax_confidence(distances, temperature=1.0):
    """
    Convert a list of distances into confidence percentages using softmax.

    Args:
        distances:   list of floats — lower = better match
        temperature: float — controls how "sharp" the distribution is.
                     Lower = more contrast between best and worst.

    Returns:
        list of floats — confidence percentages summing to ~100%
    """
    d = np.array(distances, dtype=np.float64)

    # Negate so lower distance → higher score
    neg_d = -d / temperature

    # Shift for numerical stability
    neg_d -= np.max(neg_d)

    exp_d = np.exp(neg_d)
    softmax_vals = exp_d / exp_d.sum()

    return [round(float(v) * 100, 1) for v in softmax_vals]


def rank_with_confidence(results, distance_key="final_score", top_n=3):
    """
    Take a list of match result dicts, compute softmax confidence,
    and return top-N ranked results with confidence_pct.

    Args:
        results:      list of dicts, each must have `distance_key` and `song_name`
        distance_key: which key holds the distance value
        top_n:        how many to return

    Returns:
        list of top_n dicts with added `confidence_pct`
    """
    if not results:
        return []

    # Sort ascending (lowest distance = best)
    results = sorted(results, key=lambda r: r[distance_key])

    # Extract distances for softmax
    distances = [r[distance_key] for r in results]
    confidences = softmax_confidence(distances, temperature=0.5)

    # Attach confidence to each result
    for r, conf in zip(results, confidences):
        r["confidence_pct"] = conf

    return results[:top_n]


def display_ranked(results):
    """Pretty-print ranked results for CLI."""
    print("\n" + "=" * 45)
    print("        CONFIDENCE-RANKED MATCHES")
    print("=" * 45)
    for i, r in enumerate(results):
        name = r.get("song_name", "Unknown")
        conf = r.get("confidence_pct", 0)
        tag  = "  ← 🎵 BEST" if i == 0 else ""
        print(f"  {i+1}. {name} — {conf}%{tag}")
    print("=" * 45)
