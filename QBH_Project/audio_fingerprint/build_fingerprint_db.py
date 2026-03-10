"""
audio_fingerprint/build_fingerprint_db.py — Script to generate the audfprint database.
"""

import os
import sys
import subprocess

# Ensure the project root is in sys.path so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    AUDFPRINT_SCRIPT_PATH,
    FINGERPRINT_DB_PATH,
    AUDIO_FINGERPRINT_SONGS_DIR,
    FINGERPRINT_EXTENSIONS,
)

def collect_audio_files(root_dir):
    files = []
    if not os.path.exists(root_dir):
        print(f"[ERR] Source directory not found: {root_dir}")
        return []

    for root, _, filenames in os.walk(root_dir):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in FINGERPRINT_EXTENSIONS:
                files.append(os.path.join(root, name))

    return sorted(files)

def build_db():
    os.makedirs(os.path.dirname(FINGERPRINT_DB_PATH), exist_ok=True)

    song_files = collect_audio_files(AUDIO_FINGERPRINT_SONGS_DIR)
    if not song_files:
        print(f"[ERR] No audio files found in {AUDIO_FINGERPRINT_SONGS_DIR}")
        return

    print(f"\n[fingerprint] Found {len(song_files)} files to index.")

    # Slightly denser fingerprint settings for better noisy-query matching
    cmd = [
        sys.executable,
        AUDFPRINT_SCRIPT_PATH,
        "new",
        "--dbase", FINGERPRINT_DB_PATH,
        "--density", "30",
        "--fanout", "5",
    ] + song_files

    print("\n[fingerprint] Building audfprint database...")
    print("Command snippet:", " ".join(cmd[:9]), "...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    print("\n[fingerprint] STDOUT:")
    print(result.stdout)

    print("\n[fingerprint] STDERR:")
    print(result.stderr)

    if result.returncode != 0:
        print("\n[ERR] audfprint DB build failed.")
    else:
        print(f"\n[OK] audfprint database build successful: {FINGERPRINT_DB_PATH}")

if __name__ == "__main__":
    build_db()
