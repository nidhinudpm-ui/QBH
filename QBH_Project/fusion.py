"""
fusion.py — Score fusion for QBH hybrid pipeline.

Blends melody_score and lyric_score based on query type.
  Hum:   melody_w = 1.0,  lyric_w = 0.0
  Mixed: melody_w = 0.65, lyric_w = 0.35

Outputs: melody_score, lyric_score, final_score.
confidence is NOT included — deferred to Phase 6 calibration.
"""

from config import LYRIC_WEIGHT_STRONG


def fuse_results(melody_results, lyric_scores=None, q_type="mixed"):
    if not lyric_scores:
        lyric_scores = {}

    if q_type == "hum":
        melody_w = 1.0
        lyric_w = 0.0
    else:
        melody_w = 1.0 - LYRIC_WEIGHT_STRONG
        lyric_w = LYRIC_WEIGHT_STRONG

    fused = []

    for m in melody_results:
        name = m["song_name"]
        m_score = float(m.get("melody_score", 0.0))
        l_score = float(lyric_scores.get(name, 0.0))

        final_score = (melody_w * m_score) + (lyric_w * l_score)

        entry = m.copy()
        entry["melody_score"] = m_score
        entry["lyric_score"] = l_score
        entry["final_score"] = float(final_score)
        fused.append(entry)

    fused.sort(key=lambda x: x["final_score"], reverse=True)
    return fused
