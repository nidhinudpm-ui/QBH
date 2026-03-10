"""
precache_metadata.py — Pre-fetch Spotify metadata for all songs in the database.
This populates spotify_metadata.json to ensure zero-latency lookups during queries.
"""

import os
import pickle
from config import FEATURES_PKL
from spotify_client import search_track, save_spotify_cache

def main():
    if not os.path.exists(FEATURES_PKL):
        print(f"Error: {FEATURES_PKL} not found. Run extract_features.py first.")
        return

    print("--- Pre-caching Spotify Metadata ---")
    with open(FEATURES_PKL, 'rb') as f:
        db = pickle.load(f)

    song_names = sorted(db.keys())
    print(f"Found {len(song_names)} songs in database.\n")

    count = 0
    for i, name in enumerate(song_names, 1):
        print(f"[{i}/{len(song_names)}] Processing: {name}")
        result = search_track(name)
        if result:
            print(f"  ✅ Found: {result['title']} - {result['artist']}")
            count += 1
        else:
            print(f"  ❌ Not found on Spotify")
    
    # search_track already calls save_spotify_cache internally, 
    # but let's be explicit one last time.
    save_spotify_cache()

    print(f"\nDone! Pre-cached {count}/{len(song_names)} songs in spotify_metadata.json")

if __name__ == "__main__":
    main()
