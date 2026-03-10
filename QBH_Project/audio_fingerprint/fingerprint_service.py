"""
audio_fingerprint/fingerprint_service.py — Service to identify songs via audfprint.
"""

import os
import sys
import subprocess
import tempfile

from config import AUDFPRINT_SCRIPT_PATH, FINGERPRINT_DB_PATH
from audio_fingerprint.parse_audfprint_output import parse_match_output
from audio_fingerprint.audio_utils import convert_to_wav


def identify_song_from_audio(query_audio_path: str):
    if not os.path.exists(FINGERPRINT_DB_PATH):
        return {
            "mode": "song_audio",
            "source": "audfprint",
            "matched": False,
            "error": "Fingerprint database not found. Build it first."
        }

    if not os.path.exists(query_audio_path):
        return {
            "mode": "song_audio",
            "source": "audfprint",
            "matched": False,
            "error": "Query audio file not found."
        }

    temp_wav = None

    try:
        # Always convert uploaded audio to clean mono WAV for audfprint
        fd, temp_wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        print(f"[fingerprint] Converting {query_audio_path} to {temp_wav}...")
        try:
            convert_to_wav(query_audio_path, temp_wav, sample_rate=11025)
        except Exception as e:
            return {
                "mode": "song_audio",
                "source": "audfprint",
                "matched": False,
                "error": f"Audio conversion failed: {str(e)}",
                "saved_query_file": query_audio_path
            }

        cmd = [
            sys.executable,
            AUDFPRINT_SCRIPT_PATH,
            "match",
            "--dbase", FINGERPRINT_DB_PATH,
            "--min-count", "3",
            "--exact-count",
            temp_wav
        ]

        print(f"\n[fingerprint] Executing match: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        print(f"\n[fingerprint] STDOUT:\n{result.stdout}")
        print(f"\n[fingerprint] STDERR:\n{result.stderr}")

        if result.returncode != 0:
            return {
                "mode": "song_audio",
                "source": "audfprint",
                "matched": False,
                "error": result.stderr.strip() or "audfprint match failed",
                "raw_output": result.stdout
            }

        parsed = parse_match_output(result.stdout)

        if not parsed["matched"]:
            return {
                "mode": "song_audio",
                "source": "audfprint",
                "matched": False,
                "title": None,
                "artist": None,
                "raw_output": result.stdout,
                "stderr": result.stderr,
                "parsed_debug": parsed
            }

        best = parsed["best"]
        matched_filename = os.path.basename(best["matched_file"])
        title = os.path.splitext(matched_filename)[0]

        count = best["match_count"]
        confidence_pct = min(100.0, max(0.0, count * 5.0))

        return {
            "mode": "song_audio",
            "source": "audfprint",
            "matched": True,
            "title": title.replace("_", " "),
            "song_name": matched_filename,
            "artist": "Unknown",
            "album": "Unknown",
            "confidence_pct": round(confidence_pct, 2),
            "match_count": count,
            "match_time": best["match_time"],
            "matched_file": best["matched_file"],
            "candidates": parsed["candidates"]
        }

    finally:
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except Exception:
                pass
