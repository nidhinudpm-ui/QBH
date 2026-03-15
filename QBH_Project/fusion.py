"""
fusion.py — Score fusion for QBH hybrid pipeline.

Blends melody_score and lyric_score based on query type.
  Hum:   melody_w = 1.0,  lyric_w = 0.0
  Mixed: melody_w = 0.65, lyric_w = 0.35

Outputs: melody_score, lyric_score, final_score.
confidence is NOT included — deferred to Phase 6 calibration.
"""

from config import LYRIC_WEIGHT_STRONG


def fuse_results(melody_results, lyric_scores=None, q_type="mixed", asr_conf=0.0):
    if not lyric_scores:
        lyric_scores = {}

    # Find the best lyric score to decide if it's worth using
    best_lyric = max(lyric_scores.values()) if lyric_scores else 0.0

    # Ignore lyrics entirely if ASR is garbage or the match is very trivial
    if asr_conf < 0.35 or best_lyric < 0.1:
        lyric_w = 0.0
        melody_w = 1.0
    elif q_type == "hum":
        melody_w = 0.90
        lyric_w = 0.10
    elif q_type == "mixed" and asr_conf >= 0.5:
        melody_w = 0.75
        lyric_w = 0.25
    else:  # Mixed but weak ASR
        melody_w = 0.90
        lyric_w = 0.10

    melody_map = {m["song_name"]: m for m in melody_results}
    all_song_names = set(melody_map.keys()) | set(lyric_scores.keys())
    
    fused = []
    for name in all_song_names:
        m = melody_map.get(name, {})
        m_score = float(m.get("melody_score", 0.0))
        l_score = float(lyric_scores.get(name, 0.0))

        final_score = (melody_w * m_score) + (lyric_w * l_score)

        # Preserve existing metadata from melody_results if available
        entry = m.copy() if m else {"song_name": name}
        entry["song_name"] = name
        entry["melody_score"] = m_score
        entry["lyric_score"] = l_score
        entry["final_score"] = float(final_score)
        fused.append(entry)

    fused.sort(key=lambda x: x["final_score"], reverse=True)
    return fused
