"""
config.py — Central configuration for the QBH Music Intelligence Platform
"""

import os
from dotenv import load_dotenv

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(PROJECT_DIR, ".env"))

# ─── Spotify ─────────────────────────────────────────────────────────────────
SPOTIFY_CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")

# ─── Paths ───────────────────────────────────────────────────────────────────
PARENT_DIR    = os.path.dirname(PROJECT_DIR)
SONGS_DIR     = os.path.join(PARENT_DIR, "songs")
WAV_DIR       = os.path.join(PROJECT_DIR, "wav_songs")
DATABASE_DIR  = os.path.join(PROJECT_DIR, "database")
FEATURES_PKL  = os.path.join(DATABASE_DIR, "saved_features.pkl")
UPLOAD_FOLDER = os.path.join(PROJECT_DIR, "uploads")

# ─── Audio ───────────────────────────────────────────────────────────────────
SAMPLE_RATE       = 22050
HOP_LENGTH        = 512
DOWNSAMPLE_FACTOR = 4
MAX_RECORDING_SEC = 10

# ─── Matching weights ────────────────────────────────────────────────────────
# Intervals are the only reliable signal from humming.
# Chroma is a very lightweight tie-breaker.
# Onset is NOT used — unreliable from mic recordings.
INTERVAL_WEIGHT = 0.85
CHROMA_WEIGHT   = 0.15
ONSET_WEIGHT    = 0.0   # kept for backward compatibility, not used
TOP_MATCHES     = 5
SIMILAR_DATASET = 5
