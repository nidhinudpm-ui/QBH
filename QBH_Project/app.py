"""
app.py — Flask backend for QBH Music Intelligence Platform

Endpoints:
  GET  /              → serves the recording UI
  POST /identify-song → match + confidence + metadata + recommendations

Optimizations:
  - Audio preprocessing: high-pass filter, noise reduction, pre-emphasis
  - Spotify is called ONLY for the best match (1 search + 1 top-tracks call)
  - All numpy types are sanitized to native Python before JSON serialization
  - Feature DB is preloaded at startup
"""

import os
import sys
import uuid
import pickle
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
from scipy.signal import butter, sosfilt
from flask import Flask, request, jsonify, render_template

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from config import FEATURES_PKL, UPLOAD_FOLDER, SAMPLE_RATE
from match import match_query
from spotify_client import search_track
from recommend import recommend_from_dataset, recommend_from_spotify


# ─── Numpy → Python sanitizer ────────────────────────────────────────────────
def sanitize(obj):
    """Recursively convert numpy types to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def clean_song_name(filename):
    """Turn WAV filename into a readable song name."""
    return (filename
            .replace(".wav", "")
            .replace("_spotdown.org", "")
            .replace("_", " ")
            .strip())


# ─── Audio Preprocessing Helpers ─────────────────────────────────────────────
def highpass_filter(y, sr, cutoff=80):
    """Apply high-pass Butterworth filter to remove low-frequency noise/rumble."""
    sos = butter(5, cutoff, btype='highpass', fs=sr, output='sos')
    return sosfilt(sos, y).astype(np.float32)


def pre_emphasis(y, coeff=0.97):
    """Apply pre-emphasis filter to boost high frequencies for better pitch detection."""
    return np.append(y[0], y[1:] - coeff * y[:-1]).astype(np.float32)


def preprocess_browser_audio(y, sr):
    """
    Full preprocessing chain for browser microphone recordings:
    1. High-pass filter at 80 Hz (removes rumble)
    2. Spectral noise reduction
    3. Pre-emphasis (boosts vocal frequencies)
    4. Normalize
    """
    # 1. High-pass filter
    y = highpass_filter(y, sr, cutoff=80)

    # 2. Noise reduction (spectral gating)
    y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.7, stationary=True)

    # 3. Pre-emphasis
    y = pre_emphasis(y, coeff=0.97)

    # 4. Normalize
    y = librosa.util.normalize(y)

    return y


# ─── Flask App ────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Preload feature database once at startup
_feature_db = None

def get_feature_db():
    global _feature_db
    if _feature_db is None and os.path.exists(FEATURES_PKL):
        with open(FEATURES_PKL, 'rb') as f:
            _feature_db = pickle.load(f)
        print(f"[app] Preloaded {len(_feature_db)} songs from feature DB")
    return _feature_db


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/identify-song', methods=['POST'])
def identify_song():
    """Full pipeline: preprocess → match → confidence → Spotify → recommendations."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']

    try:
        tid      = str(uuid.uuid4())[:8]
        raw_path = os.path.join(UPLOAD_FOLDER, f"{tid}_raw.wav")
        wav_path = os.path.join(UPLOAD_FOLDER, f"{tid}_query.wav")

        audio_file.save(raw_path)
        print(f"[identify] Received {os.path.getsize(raw_path)} bytes")

        # ── Step 1: Load + preprocess audio with noise reduction ──
        y, sr = librosa.load(raw_path, sr=SAMPLE_RATE, mono=True)
        y = preprocess_browser_audio(y, sr)
        sf.write(wav_path, y, SAMPLE_RATE)

        # ── Step 2: Match with softmax confidence ──
        top_matches = match_query(
            query_file=wav_path, pkl_path=FEATURES_PKL,
            top_n=5, return_results=True
        )

        # Cleanup temp files
        for p in [raw_path, wav_path]:
            if os.path.exists(p):
                os.remove(p)

        if not top_matches:
            return jsonify({'error': 'Could not identify. Hum louder or longer.'}), 500

        best = top_matches[0]
        best_song_name = best["song_name"]

        # ── Step 3: Format top matches (NO Spotify calls — fast) ──
        top_matches_out = []
        for m in top_matches:
            top_matches_out.append({
                "title":      clean_song_name(m["song_name"]),
                "confidence": float(m.get("confidence_pct", 0))
            })

        # ── Step 4: Dataset recommendations (NO Spotify — fast) ──
        similar_dataset = recommend_from_dataset(best_song_name)
        similar_dataset_out = []
        for s in similar_dataset:
            similar_dataset_out.append({
                "title":      clean_song_name(s["song_name"]),
                "similarity": float(s["similarity"])
            })

        # ── Step 5: Spotify — ONLY for the best match (1 API call) ──
        spotify_meta = search_track(best_song_name)

        identified = {
            "title":        spotify_meta["title"] if spotify_meta else clean_song_name(best_song_name),
            "artist":       spotify_meta.get("artist", "") if spotify_meta else "",
            "album":        spotify_meta.get("album", "") if spotify_meta else "",
            "release_date": spotify_meta.get("release_date", "") if spotify_meta else "",
            "confidence":   float(best.get("confidence_pct", 0)),
            "preview_url":  spotify_meta.get("preview_url", "") if spotify_meta else "",
            "image":        spotify_meta.get("image", "") if spotify_meta else "",
            "spotify_url":  spotify_meta.get("spotify_url", "") if spotify_meta else ""
        }

        # ── Step 6: Spotify artist recs (1 more API call) ──
        artist_id = spotify_meta.get("artist_id", "") if spotify_meta else ""
        similar_spotify = recommend_from_spotify(artist_id, limit=5)

        # ── Build response & sanitize ALL numpy types ──
        response = sanitize({
            "success":               True,
            "identified_song":       identified,
            "top_matches":           top_matches_out,
            "similar_songs_dataset": similar_dataset_out,
            "similar_songs_spotify": similar_spotify
        })

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 48)
    print("  🎵 QBH Music Intelligence Platform")
    print("=" * 48)

    db = get_feature_db()
    if db:
        print(f"  ✅ Feature DB: {len(db)} songs")
    else:
        print("  ⚠️  Feature DB not found! Run: python extract_features.py")

    print(f"  🌐 Open http://localhost:5000\n")
    app.run(debug=True, port=5000)
