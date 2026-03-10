"""
audio_fingerprint/parse_audfprint_output.py — Strict parser for audfprint match output.
"""

import os
import re
from config import AUDIO_FINGERPRINT_SONGS_DIR


def is_from_dataset(path_str: str) -> bool:
    if not path_str:
        return False

    dataset_root = os.path.normcase(os.path.abspath(AUDIO_FINGERPRINT_SONGS_DIR))
    candidate = os.path.normcase(os.path.abspath(path_str))
    return candidate.startswith(dataset_root)


def parse_match_output(stdout_text: str):
    lines = [line.strip() for line in stdout_text.splitlines() if line.strip()]
    candidates = []

    for line in lines:
        parts = line.split()

        # Format 1: query_file hit_rank hit_file match_count match_time
        if len(parts) >= 5:
            try:
                query_file = parts[0]
                hit_rank = int(parts[1])
                hit_file = parts[2]
                match_count = float(parts[3])
                match_time = float(parts[4])

                if not is_from_dataset(hit_file):
                    continue

                candidates.append({
                    "query_file": query_file,
                    "rank": hit_rank,
                    "matched_file": hit_file,
                    "match_count": match_count,
                    "match_time": match_time,
                    "raw_line": line,
                })
                continue
            except (ValueError, IndexError):
                pass

        # Format 2: Matched [query] to [target] with [N] hits at [T]
        m = re.search(
            r"Matched\s+(.*?)\s+to\s+(.*?)\s+with\s+([\d.]+)\s+hits?\s+at\s+([\d.]+)",
            line,
            re.IGNORECASE
        )
        if m:
            hit_file = m.group(2)

            if not is_from_dataset(hit_file):
                continue

            candidates.append({
                "query_file": m.group(1),
                "rank": len(candidates) + 1,
                "matched_file": hit_file,
                "match_count": float(m.group(3)),
                "match_time": float(m.group(4)),
                "raw_line": line,
            })

    candidates.sort(key=lambda x: x["match_count"], reverse=True)
    best = candidates[0] if candidates else None

    return {
        "matched": best is not None,
        "best": best,
        "candidates": candidates,
        "raw_lines": lines,
    }
