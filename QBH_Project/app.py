"""
app.py — Flask backend for QBH Music Intelligence Platform (Stabilized)
"""

import os
import sys
import uuid
import json
import pickle
import librosa
import numpy as np
import soundfile as sf
import time
from flask import Flask, request, jsonify, render_template

# ─── Live Debug Logger (Phase 9) ──────────────────────────────────────────────
class Logger(object):
    def __init__(self, filename="debug_live.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()
sys.stderr = sys.stdout

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from config import FEATURES_PKL, UPLOAD_FOLDER, SAMPLE_RATE, DATABASE_DIR
from match import match_query
from extract_features import print_segment_stats
from feedback_store import log_feedback
from spotify_client import search_track, get_track_details, get_similar_tracks, get_youtube_search_url, load_spotify_cache
from recommend import recommend_from_dataset, recommend_from_spotify

# ─── Song Details JSON Cache ─────────────────────────────────────────────────
SONG_DETAILS_CACHE = os.path.join(DATABASE_DIR, "song_details_cache.json")

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

def clean_song_name(name):
    """Turn WAV filename into a readable song name."""
    return (name
            .replace(".wav", "")
            .replace("_spotdown.org", "")
            .replace("_", " ")
            .strip())

from flask_cors import CORS

# ─── Flask App Setup ─────────────────────────────────────────────────────────
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app) # Allow all origins for troubleshooting
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Global Settings ──────────────────────────────────────────────────────────
DEBUG_MELODY_ONLY = True  # Set to False to re-enable Spotify/Lyrics enrichment

# ── Global Feature DB Preload ──────────────────────────────────────────────
FEATURE_DB = {}

def preload_db():
    global FEATURE_DB
    if os.path.exists(FEATURES_PKL):
        print(f"\n[app] Preloading {os.path.basename(FEATURES_PKL)}...", flush=True)
        try:
            with open(FEATURES_PKL, 'rb') as f:
                FEATURE_DB = pickle.load(f)
            print(f"  [OK] Feature DB: {len(FEATURE_DB)} songs loaded into memory", flush=True)
            print_segment_stats(FEATURE_DB)
        except Exception as e:
            print(f"  [ERR] DB preload failed: {e}", flush=True)

preload_db()

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/ping')
def ping():
    return jsonify({"success": True, "message": "Backend is reachable!"})

@app.route('/')
def index():
    from config import RETRY_ACCUMULATE_EXCLUSIONS
    return render_template('index.html', RETRY_ACCUMULATE_EXCLUSIONS=RETRY_ACCUMULATE_EXCLUSIONS)

@app.route('/identify-song', methods=['POST'])
def identify_song():
    """
    QBH Pipeline Endpoint.
    Default: Fast melody-only path if DEBUG_MELODY_ONLY = True.
    """
    t_start_total = time.time()
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    tid = str(uuid.uuid4())[:8]
    raw_path = os.path.join(UPLOAD_FOLDER, f"{tid}_raw.wav")
    wav_path = os.path.join(UPLOAD_FOLDER, f"{tid}_query.wav")
    
    audio_file.save(raw_path)

    try:
        # Notes: formatting wav path for backend
        if os.path.exists(wav_path): os.remove(wav_path)
        os.rename(raw_path, wav_path)

        print(f"\n[app] ID Request: {audio_file.filename} (debug={DEBUG_MELODY_ONLY})", flush=True)
        
        # ── Step 1: Matching ──
        results = match_query(
            query_file=wav_path, db=FEATURE_DB, 
            top_n=5, return_results=True,
            debug_only=DEBUG_MELODY_ONLY
        )

        # Cleanup
        if os.path.exists(wav_path): os.remove(wav_path)

        if not results:
            return jsonify({'error': 'No melody detected. Try humming louder.'}), 200

        best = results[0]
        
        # Hybrid Enrichment (Phase 12: Rich Metadata for ALL)
        top_matches_out = []
        internal_names = []
        for r in results:
            s_name = r["song_name"]
            internal_names.append(s_name)
            
            # Fetch Spotify metadata (Cached)
            sp = search_track(s_name) or {}
            
            top_matches_out.append({
                "title":           sp.get("title") or clean_song_name(s_name),
                "song_name":       s_name,
                "internal_name":   s_name,
                "artist":          sp.get("artist") or r.get("artist", "Unknown"),
                "album":           sp.get("album") or r.get("album", "Unknown"),
                "release_date":    sp.get("release_date") or r.get("release_date", "—"),
                "image":           sp.get("image") or r.get("image", ""),
                "spotify_url":     sp.get("spotify_url") or r.get("spotify_url", ""),
                "preview_url":     sp.get("preview_url") or r.get("preview_url", ""),
                "final_score":     r["final_score"],
                "melody_score":    r["melody_score"],
                "lyric_score":     r.get("lyric_score", 0),
                "confidence_pct":  r["confidence_pct"],
                "confidence":      r["confidence_pct"],  # Backward compatibility
                "waveform":        r.get("waveform", {}),
                "debug":           r.get("debug", {})
            })

        resp = sanitize({
            "success":         True,
            "query_id":        tid,
            "identified_song": top_matches_out[0] if top_matches_out else None,
            "top_matches":     top_matches_out,
            "internal_names":  internal_names,
            "q_type":          results[0]["q_type"] if results else "mixed",
            "debug":           results[0]["debug"] if results else {},
            "similar_songs_dataset": [],
            "similar_songs_spotify": []
        })
        
        # Summary Logging (Phase 11)
        print("\n[app] Outgoing Matches:", flush=True)
        for m in top_matches_out[:3]:
            print(f"  - {m['internal_name']:30s} | Conf: {m['confidence_pct']:4.1f}% | Mel: {m['melody_score']:.3f} | WF: {m['waveform']['status']}", flush=True)
            
        print(f"[app] Total Request Time: {time.time()-t_start_total:.2f}s", flush=True)
        return jsonify(resp)

    except Exception as e:
        print(f"  ❌ Endpoint Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ─── Feedback Endpoints ──────────────────────────────────────────────────────

MAX_RETRY_DEPTH = 2  # Max times user can re-hum in one session

@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    """
    POST /submit-feedback
    Body: { query_id, shown_list, selected_song, mode, q_type?,
            retry_depth?, excluded_songs?, melody_score?, top1_score? }
    Modes: selected_from_list | close_but_wrong | retry_excluding_previous
    """
    body = request.get_json(silent=True) or {}
    query_id     = body.get("query_id", "?")
    shown_list   = body.get("shown_list", [])
    selected     = body.get("selected_song", "")
    mode         = body.get("mode", "")
    retry_depth  = int(body.get("retry_depth", 0))

    try:
        from config import QUERY_PITCH_BACKEND
        entry = {
            "query_id":           query_id,
            "mode":               mode,
            "shown_list":         shown_list,
            "selected_song":      selected,
            "selected_rank":      body.get("selected_rank", -1),
            "q_type":             body.get("q_type", ""),
            "pitch_backend_used": QUERY_PITCH_BACKEND,
            "selected_melody_score": float(body.get("melody_score", 0)),
            "top1_final_score":   float(body.get("top1_score", 0)),
            "excluded_songs":     body.get("excluded_songs", []),
            "retry_depth":        retry_depth,
        }
        log_feedback(entry)
        return jsonify({"success": True, "message": "Feedback recorded. Thank you!"})
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        print(f"[app] Feedback error: {e}", flush=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/identify-song-retry', methods=['POST'])
def identify_song_retry():
    """
    POST /identify-song-retry
    FormData: audio file + excluded_songs (JSON array) + retry_depth (int)
    Runs match_query excluding previously shown songs.
    """
    t_start = time.time()

    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400

    retry_depth = int(request.form.get("retry_depth", 1))
    if retry_depth > MAX_RETRY_DEPTH:
        return jsonify({'error': f'Max retry depth ({MAX_RETRY_DEPTH}) reached. Please start fresh.'}), 400

    excluded_songs = []
    try:
        excluded_songs = json.loads(request.form.get("excluded_songs", "[]"))
    except Exception:
        pass

    audio_file = request.files['audio']
    tid = str(uuid.uuid4())[:8]
    raw_path = os.path.join(UPLOAD_FOLDER, f"{tid}_raw.wav")
    wav_path = os.path.join(UPLOAD_FOLDER, f"{tid}_query.wav")
    audio_file.save(raw_path)

    try:
        if os.path.exists(wav_path): os.remove(wav_path)
        os.rename(raw_path, wav_path)

        from config import RETRY_ACCUMULATE_EXCLUSIONS
        print(f"\n[app] Retry Request: id={tid} depth={retry_depth} accumulate={RETRY_ACCUMULATE_EXCLUSIONS}", flush=True)
        print(f"[app] Excluded count={len(excluded_songs)} names={excluded_songs}", flush=True)

        results = match_query(
            query_file=wav_path, db=FEATURE_DB,
            top_n=5, return_results=True,
            debug_only=DEBUG_MELODY_ONLY,
            excluded_songs=excluded_songs
        )

        if os.path.exists(wav_path): os.remove(wav_path)

        if not results:
            return jsonify({'error': 'No melody detected. Try humming louder.'}), 200

        # Phase 12 Enrichment
        top_matches_out = []
        internal_names = []
        for r in results:
            s_name = r["song_name"]
            internal_names.append(s_name)
            sp = search_track(s_name) or {}
            top_matches_out.append({
                "title":         sp.get("title") or clean_song_name(s_name),
                "song_name":     s_name,
                "internal_name": s_name,
                "artist":        sp.get("artist") or "Unknown",
                "album":         sp.get("album") or "Unknown",
                "image":         sp.get("image") or "",
                "spotify_url":   sp.get("spotify_url") or "",
                "final_score":   r["final_score"],
                "melody_score":  r["melody_score"],
                "lyric_score":   r["lyric_score"],
                "confidence_pct": r["confidence_pct"],
                "confidence":     r["confidence_pct"],
                "waveform":      r.get("waveform", {}),
                "debug":         r.get("debug", {})
            })

        resp = sanitize({
            "success":         True,
            "query_id":        tid,
            "retry_depth":     retry_depth,
            "identified_song": top_matches_out[0],
            "top_matches":     top_matches_out,
            "internal_names":  internal_names,
            "q_type":          results[0]["q_type"] if results else "mixed",
            "debug":           results[0]["debug"] if results else {}
        })
        print(f"[app] Retry Total Time: {time.time()-t_start:.2f}s", flush=True)
        return jsonify(resp)

    except Exception as e:
        print(f"[app] Retry Error: {e}", flush=True)
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/song-details', methods=['POST'])
def song_details_post():
    """
    Fetch full details for a song: metadata, links, similar tracks.
    """
    body = request.get_json(silent=True) or {}
    song_name = body.get("song_name", "").strip()
    if not song_name:
        return jsonify({"error": "song_name is required"}), 400

    # Search Spotify via client (uses internal caching)
    meta = search_track(song_name)
    if not meta:
        return jsonify({"error": "Song not found on Spotify"}), 404

    track_id = meta.get("track_id", "")
    details = get_track_details(track_id) or meta
    similar = get_similar_tracks(track_id, limit=5)
    youtube_url = get_youtube_search_url(
        details.get("title", ""), details.get("artist", "")
    )

    dur_ms = details.get("duration_ms", 0)
    mins, secs = divmod(dur_ms // 1000, 60)
    duration_str = f"{mins}:{secs:02d}"

    result = {
        "title":        details.get("title", ""),
        "artist":       details.get("artist", ""),
        "all_artists":  details.get("all_artists", details.get("artist", "")),
        "album":        details.get("album", ""),
        "album_type":   details.get("album_type", ""),
        "release_date": details.get("release_date", ""),
        "duration":     duration_str,
        "popularity":   details.get("popularity", 0),
        "explicit":     details.get("explicit", False),
        "image":        details.get("image", ""),
        "spotify_url":  details.get("spotify_url", ""),
        "youtube_url":  youtube_url,
        "preview_url":  details.get("preview_url", ""),
        "similar_tracks": similar
    }

    return jsonify(sanitize(result))

# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    load_spotify_cache()
    
    print("=" * 48)
    print("  🎵 QBH Music Intelligence Platform (Stabilized)")
    print("=" * 48)
    print(f"  ✅ Feature DB: {len(FEATURE_DB)} songs")
    print(f"  🌐 Server running on http://0.0.0.0:5000", flush=True)
    
    app.run(host='0.0.0.0', port=5000, debug=False)
