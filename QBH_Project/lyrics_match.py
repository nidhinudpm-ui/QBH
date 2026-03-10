"""
lyrics_match.py — Simplified ASR-based lyric matching for QBH reranking.

Phase 10: Replaced embedding-based matching with rapidfuzz title matching.

Pipeline:
  1. Transcribe query audio with Whisper (if singing detected)
  2. For each candidate song, compute:
     a. Title match: rapidfuzz token_set_ratio(transcript, song_title)
     b. Phrase match: rapidfuzz partial_ratio(transcript, lyric_phrases) (if metadata exists)
  3. Return scored dict per song

Title matching is the primary and fastest path.
Phrase matching is a secondary boost if lyrics_metadata.json exists.
"""

import os
import sys
import json

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from config import (
    WHISPER_MODEL, PREFER_FASTER_WHISPER,
    LYRIC_MIN_CONF, LYRIC_FUZZY_THRESHOLD
)

LYRICS_METADATA = os.path.join(PROJECT_DIR, "lyrics_metadata.json")

_lyrics_db = None
_asr_model = None


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _clean_song_title(filename: str) -> str:
    """Extract readable title from a song filename."""
    return (filename
            .replace(".wav", "")
            .replace("_spotdown.org", "")
            .replace("_", " ")
            .strip()
            .lower())


def _load_lyrics_db():
    """Load lyrics metadata (cached). Optional — title matching works without it."""
    global _lyrics_db
    if _lyrics_db is None:
        if os.path.exists(LYRICS_METADATA):
            with open(LYRICS_METADATA, 'r', encoding='utf-8') as f:
                _lyrics_db = json.load(f)
        else:
            _lyrics_db = {}
    return _lyrics_db


def _get_asr_model():
    """Lazy-load the ASR model (faster-whisper preferred)."""
    global _asr_model
    if _asr_model is not None:
        return _asr_model

    if PREFER_FASTER_WHISPER:
        try:
            from faster_whisper import WhisperModel
            _asr_model = ("faster_whisper", WhisperModel(WHISPER_MODEL, compute_type="int8"))
            print(f"[lyrics] Loaded faster-whisper ({WHISPER_MODEL})", flush=True)
            return _asr_model
        except ImportError:
            print("[lyrics] faster-whisper not available, trying whisper...", flush=True)
        except Exception as e:
            print(f"[lyrics] faster-whisper failed: {e}, trying whisper...", flush=True)

    try:
        import whisper
        _asr_model = ("whisper", whisper.load_model(WHISPER_MODEL))
        print(f"[lyrics] Loaded whisper ({WHISPER_MODEL})", flush=True)
        return _asr_model
    except ImportError:
        print("[lyrics] WARNING: No ASR engine found. Lyric branch disabled.", flush=True)
        _asr_model = ("none", None)
        return _asr_model


# ─── Transcription ─────────────────────────────────────────────────────────────

def transcribe_query(audio_path):
    """
    Transcribe query audio to text.
    Returns (transcript: str, confidence: float) or (None, 0.0).
    """
    model_type, model = _get_asr_model()
    if model_type == "none" or model is None:
        return None, 0.0

    try:
        import numpy as np
        if model_type == "faster_whisper":
            segments, info = model.transcribe(
                audio_path, beam_size=3, language=None, vad_filter=True
            )
            text_parts, conf_parts = [], []
            for seg in segments:
                text_parts.append(seg.text.strip())
                conf_parts.append(seg.avg_logprob)
            transcript = " ".join(text_parts).strip().lower()
            avg_conf = float(np.exp(np.mean(conf_parts))) if conf_parts else 0.0
            return transcript, avg_conf

        elif model_type == "whisper":
            result = model.transcribe(audio_path, fp16=False)
            transcript = result.get("text", "").strip().lower()
            segs = result.get("segments", [])
            avg_conf = float(1.0 - np.mean([s.get("no_speech_prob", 0.5) for s in segs])) if segs else 0.0
            return transcript, avg_conf

    except Exception as e:
        print(f"[lyrics] Transcription failed: {e}", flush=True)
    return None, 0.0


# ─── Matching ──────────────────────────────────────────────────────────────────

def _rapidfuzz_score(a: str, b: str) -> float:
    """
    Compute similarity between two strings using rapidfuzz.
    Returns [0.0, 1.0].
    Falls back to simple substring check if rapidfuzz is unavailable.
    """
    try:
        from rapidfuzz import fuzz
        # token_set_ratio handles word-order variation well (e.g., "aluva puzha" vs "puzha aluva")
        return fuzz.token_set_ratio(a, b) / 100.0
    except ImportError:
        return 1.0 if b in a else 0.0


def match_by_title(transcript: str, candidate_songs: list) -> dict:
    """
    Primary matching: compare ASR transcript to song title using rapidfuzz.

    Score mapping:
      >= 0.85  →  1.0  (strong title match)
      >= 0.65  →  0.6  (partial match)
      else     →  0.0
    """
    scores = {}
    for song_file in candidate_songs:
        title = _clean_song_title(song_file)
        ratio = _rapidfuzz_score(transcript, title)
        if ratio >= 0.85:
            scores[song_file] = 1.0
        elif ratio >= 0.65:
            scores[song_file] = 0.6
        else:
            scores[song_file] = 0.0
    return scores


def match_by_phrases(transcript: str, candidate_songs: list) -> dict:
    """
    Secondary matching: compare to known lyric phrases (requires lyrics_metadata.json).
    Returns a score in [0, 1] per song.
    """
    lyrics_db = _load_lyrics_db()
    scores = {}
    for song_file in candidate_songs:
        meta = lyrics_db.get(song_file, {})
        phrases = meta.get("phrases", []) + meta.get("romanized_phrases", [])
        if not phrases:
            scores[song_file] = 0.0
            continue
        best = max(_rapidfuzz_score(transcript, p.lower()) for p in phrases)
        scores[song_file] = best if best >= LYRIC_FUZZY_THRESHOLD else 0.0
    return scores


# ─── Main entry point ─────────────────────────────────────────────────────────

def transcribe_and_match(audio_path, candidate_songs):
    """
    Full lyric matching pipeline.

    Args:
        audio_path: path to query WAV
        candidate_songs: list of song filenames (from melody top-N)

    Returns:
        (lyric_scores: dict, transcript: str, asr_confidence: float)
    """
    transcript, asr_conf = transcribe_query(audio_path)
    print(f"[lyrics] Transcript: '{transcript}' (conf={asr_conf:.2f})", flush=True)

    if not transcript or asr_conf < LYRIC_MIN_CONF:
        print("[lyrics] Low confidence or empty transcript — skipping.", flush=True)
        return {}, transcript, asr_conf

    # Primary: title matching (always runs)
    title_scores = match_by_title(transcript, candidate_songs)

    # Secondary: phrase matching (runs only if metadata exists)
    phrase_scores = match_by_phrases(transcript, candidate_songs)

    # Combine: max of title and phrase score per song
    combined = {}
    for song in candidate_songs:
        combined[song] = max(title_scores.get(song, 0.0), phrase_scores.get(song, 0.0))

    # Log non-zero scores
    for song, score in sorted(combined.items(), key=lambda x: -x[1]):
        if score > 0:
            print(f"  [lyric] {song[:40]:40s}: {score:.3f}", flush=True)

    return combined, transcript, asr_conf
