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


def recommend_from_dataset(identified_song_name, pkl_path=FEATURES_PKL, top_n=SIMILAR_DATASET):
    """
    Find songs most similar to the identified song using chroma cosine similarity.
    Excludes the identified song itself.

    Returns list of dicts: [{ song_name, similarity }, ...]
    """
    if not os.path.exists(pkl_path):
        return []

    with open(pkl_path, 'rb') as f:
        db = pickle.load(f)

    if identified_song_name not in db:
        return []

    target_chroma = db[identified_song_name]["chroma"]
    similarities  = []

    for song_name, feats in db.items():
        if song_name == identified_song_name:
            continue

        cos_dist = cosine(target_chroma, feats["chroma"])
        sim      = max(0.0, 1.0 - cos_dist)

        similarities.append({
            "song_name":  song_name,
            "similarity": round(sim * 100, 1)
        })

    # Sort descending by similarity
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return similarities[:top_n]


def recommend_from_spotify(artist_id, limit=5):
    """
    Get similar songs from Spotify using the artist's top tracks.

    Returns list of dicts: [{ title, artist, album, preview_url, image }, ...]
    """
    if not artist_id:
        return []

    return get_artist_top_tracks(artist_id, limit=limit)
