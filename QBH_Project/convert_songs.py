"""
convert_songs.py — Convert MP3 songs to WAV (16kHz mono) for QBH feature extraction.

Usage:
    python convert_songs.py
"""

import os
import sys
import librosa
import soundfile as sf

# Project config
SAMPLE_RATE = 16000
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SONGS_DIR = os.path.join(PARENT_DIR, "songs")


def convert_mp3_to_wav(songs_dir=SONGS_DIR, sr=SAMPLE_RATE):
    """Convert all MP3 files in songs_dir to 16kHz mono WAV."""
    if not os.path.exists(songs_dir):
        print(f"Error: '{songs_dir}' not found.")
        return

    mp3_files = sorted(f for f in os.listdir(songs_dir) if f.lower().endswith('.mp3'))
    if not mp3_files:
        print(f"No MP3 files found in '{songs_dir}'.")
        return

    print(f"\n{'='*60}")
    print(f"  Converting {len(mp3_files)} MP3 files to WAV ({sr}Hz mono)")
    print(f"  Source: {songs_dir}")
    print(f"{'='*60}\n")

    success = 0
    for i, mp3_file in enumerate(mp3_files, 1):
        mp3_path = os.path.join(songs_dir, mp3_file)
        wav_file = os.path.splitext(mp3_file)[0] + ".wav"
        wav_path = os.path.join(songs_dir, wav_file)

        if os.path.exists(wav_path):
            print(f"[{i}/{len(mp3_files)}] {wav_file[:55]:55s} EXISTS (skip)")
            success += 1
            continue

        try:
            y, _ = librosa.load(mp3_path, sr=sr, mono=True)
            sf.write(wav_path, y, sr, subtype='PCM_16')
            duration = len(y) / sr
            print(f"[{i}/{len(mp3_files)}] {wav_file[:55]:55s} OK ({duration:.1f}s)")
            success += 1
        except Exception as e:
            print(f"[{i}/{len(mp3_files)}] {mp3_file[:55]:55s} ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"  Done! {success}/{len(mp3_files)} converted successfully.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    convert_mp3_to_wav()
