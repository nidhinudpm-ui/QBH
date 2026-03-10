"""
config.py — Central configuration for the Hybrid QBH Platform
"""

import os
from dotenv import load_dotenv

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(PROJECT_DIR, ".env"))

# --- Spotify ---
SPOTIFY_CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")

# --- Paths ---
PARENT_DIR    = os.path.dirname(PROJECT_DIR)
SONGS_DIR     = os.path.join(PARENT_DIR, "songs")
WAV_DIR       = SONGS_DIR
DATABASE_DIR  = os.path.join(PROJECT_DIR, "database")
FEATURES_PKL  = os.path.join(DATABASE_DIR, "saved_features.pkl")
UPLOAD_FOLDER = os.path.join(PROJECT_DIR, "uploads")

# --- Audio Preprocessing ---
SAMPLE_RATE       = 16000
MAX_RECORDING_SEC = 15   # Restored from 10
HPF_CUTOFF        = 100

# --- Basic Pitch (Fallback) ---
CONFIDENCE_THRESHOLD = 0.6
MIN_NOTE_DURATION    = 0.08

# --- Query F0 Extraction (pYIN) ---
PYIN_FMIN = "C2"
PYIN_FMAX = "C7"
F0_HOP_LENGTH = 256
F0_SMOOTH_KERNEL = 5
MIN_VOICED_FRAMES = 20

# --- Query Pitch Backend (A/B Testing) ---
QUERY_PITCH_BACKEND = "pyin"       # "pyin" | "torchcrepe"
RETRY_ACCUMULATE_EXCLUSIONS = True # Union vs immediate list only

# torchcrepe settings (only used when QUERY_PITCH_BACKEND = "torchcrepe")
TORCHCREPE_SAMPLE_RATE         = 16000
TORCHCREPE_HOP_LENGTH          = 256
TORCHCREPE_MODEL               = "tiny"   # "tiny" (fast) or "full" (precise)
TORCHCREPE_FMIN                = 65.4     # C2 Hz
TORCHCREPE_FMAX                = 1046.5   # C6 Hz
TORCHCREPE_BATCH_SIZE          = 512
TORCHCREPE_VOICING_THRESHOLD   = 0.21
TORCHCREPE_USE_GPU_IF_AVAILABLE = True

# --- Contour Segmentation ---
SEGMENT_LENGTH_SEC = 6
SEGMENT_OVERLAP = 0.5

# --- Matching ---
TOP_MATCHES        = 5
FAST_DTW_RADIUS   = 5
LEN_RATIO_MIN      = 0.4
LEN_RATIO_MAX      = 2.5
PRE_FILTER_HIST_N  = 20      
HIST_REJECT_THRESHOLD = 0.95  
DDTW_EARLY_EXIT_THRESH = 100.0 
MAX_SEGMENTS_PER_SONG = 5      
LOW_INFO_THRESHOLD = 0.2
SIMILAR_DATASET    = 5    

# --- Melody Weights (Targeted Tuning) ---
W_INTERVAL_HIST = 0.30
W_INTERVAL_DDTW = 0.35
W_CONTOUR_DTW   = 0.30
W_RHYTHM        = 0.05

# --- Lyrics ---
ENABLE_LYRICS         = True
LYRIC_RERANK_N        = 10   # Only rerank top 10 melody hits
LYRIC_MIN_CONF        = 0.25
LYRIC_FUZZY_THRESHOLD = 0.55
LYRIC_WEIGHT_WEAK     = 0.15
LYRIC_WEIGHT_STRONG    = 0.35
WHISPER_MODEL         = "base" # or "base"
PREFER_FASTER_WHISPER = True
