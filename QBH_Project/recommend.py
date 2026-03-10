"""
recommend.py — Song recommendation engine

Two methods:
  A) Dataset-based: chroma cosine similarity from saved_features.pkl
  B) Spotify-based:  artist's top tracks via Spotify API
"""

import os
import pickle
import numpy as np
from scipy.spatial.distance import cosine
from config import FEATURES_PKL, SIMILAR_DATASET
from spotify_client import get_artist_top_tracks


def recommend_from_dataset(identified_song_name, artist_name=None, pkl_path=FEATURES_PKL, top_n=SIMILAR_DATASET):
    """
    Find songs most similar to the identified song using chroma cosine similarity.
    Fallback: If identified_song is missing, try to find other songs by the same artist in the DB.
    """
    if not os.path.exists(pkl_path):
        return []

    with open(pkl_path, 'rb') as f:
        db = pickle.load(f)

    target_song_key = None
    if identified_song_name in db:
        target_song_key = identified_song_name
    else:
        # Try variants
        variants = [identified_song_name + ".wav", identified_song_name.replace(".wav", "")]
        for v in variants:
            if v in db:
                target_song_key = v
                break

    if target_song_key and "chroma" in db[target_song_key]:
        target_chroma = db[target_song_key]["chroma"]
        similarities  = []
        for song_name, feats in db.items():
            if song_name == target_song_key: continue
            if "chroma" not in feats: continue
            cos_dist = cosine(target_chroma, feats["chroma"])
            sim = max(0.0, 1.0 - cos_dist)
            similarities.append({
                "song_name":  song_name,
                "similarity": round(sim * 100, 1)
            })
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_n]

    # Fallback: Find other songs by artist in the DB
    if artist_name:
        artist_name_lower = artist_name.lower()
        artist_matches = []
        for song_name_db, feats in db.items():
            if song_name_db == target_song_key: continue
            # Check if internal name contains artist or part of artist
            if artist_name_lower in song_name_db.lower():
                artist_matches.append({
                    "song_name": song_name_db,
                    "similarity": 95.0 # Virtual score for UI sorting
                })
        if artist_matches:
            # Sort by variant match or just return
            return artist_matches[:top_n]

    # Final fallback: just top songs from DB (limit to ensure variety)
    fallback = []
    for k in list(db.keys()):
        if k == target_song_key: continue
        fallback.append({"song_name": k, "similarity": 40.0})
        if len(fallback) >= top_n: break
    return fallback


def recommend_from_spotify(artist_id, limit=5):
    """
    Get similar songs from Spotify using the artist's top tracks.

    Returns list of dicts: [{ title, artist, album, preview_url, image }, ...]
    """
    if not artist_id:
        return []

    return get_artist_top_tracks(artist_id, limit=limit)
