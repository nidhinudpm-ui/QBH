"""
spotify_client.py — Spotify Web API integration

Uses Client Credentials flow (no user login needed).
Credentials are loaded securely from .env via config.py.
Token is cached and reused until expiry.
"""

import time
import json
import os
import requests
from difflib import SequenceMatcher
from urllib.parse import quote_plus
from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, DATABASE_DIR

# ─── Cache Configuration ──────────────────────────────────────────────────────
SPOTIFY_CACHE_FILE = os.path.join(DATABASE_DIR, "spotify_metadata.json")
_token_cache = {"access_token": None, "expires_at": 0}
_metadata_cache = {} # In-memory cache for this session

SPOTIFY_TOKEN_URL  = "https://accounts.spotify.com/api/token"
SPOTIFY_SEARCH_URL = "https://api.spotify.com/v1/search"
SPOTIFY_ARTISTS_URL = "https://api.spotify.com/v1/artists"


def normalize_song_name(name):
    """
    Standardize song names for consistent cache lookups.
    Lowercase, trim, remove extension/suffixes, and replace underscores.
    """
    if not name:
        return ""
    # 1. Lowercase and trim
    s = name.lower().strip()
    # 2. Remove common extensions/artifacts
    for artifact in [".wav", "_spotdown.org", "(from ", "from _"]:
        s = s.replace(artifact, "")
    # 3. Replace underscores/dashes with spaces for better searching
    s = s.replace("_", " ").replace("-", " ")
    # 4. Collapse multiple spaces
    s = " ".join(s.split())
    return s


def load_spotify_cache():
    """Load persistent metadata from disk into the in-memory cache."""
    global _metadata_cache
    if os.path.exists(SPOTIFY_CACHE_FILE):
        try:
            with open(SPOTIFY_CACHE_FILE, 'r', encoding='utf-8') as f:
                _metadata_cache.update(json.load(f))
            print(f"[spotify] Loaded {len(_metadata_cache)} cached songs from disk")
        except Exception as e:
            print(f"[spotify] Cache load error: {e}")
    return _metadata_cache


def save_spotify_cache():
    """Save the in-memory metadata cache back to disk."""
    try:
        os.makedirs(os.path.dirname(SPOTIFY_CACHE_FILE), exist_ok=True)
        with open(SPOTIFY_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(_metadata_cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[spotify] Cache save error: {e}")


# Initialize cache immediately on import
load_spotify_cache()


def get_access_token():
    """
    Get a Spotify access token using Client Credentials flow.
    Caches the token and reuses until it expires.
    """
    global _token_cache

    # Return cached token if still valid (with 60s buffer)
    if _token_cache["access_token"] and time.time() < _token_cache["expires_at"] - 60:
        return _token_cache["access_token"]

    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        print("[spotify] WARNING: Missing credentials in .env file")
        return None

    try:
        response = requests.post(
            SPOTIFY_TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        _token_cache["access_token"] = data["access_token"]
        _token_cache["expires_at"]   = time.time() + data.get("expires_in", 3600)

        print("[spotify] Token acquired successfully")
        return _token_cache["access_token"]

    except requests.RequestException as e:
        print(f"[spotify] Token error: {e}")
        return None


def _auth_headers():
    """Return authorization headers with cached token."""
    token = get_access_token()
    if not token:
        return None
    return {"Authorization": f"Bearer {token}"}


def search_track(song_name):
    """
    Search Spotify for a track by name.
    Uses normalized key caching to avoid redundant API calls.
    Returns dict with: title, artist, artist_id, album, release_date, image, preview_url
    """
    if not song_name:
        return None

    norm_name = normalize_song_name(song_name)
    
    # 1. Check Cache
    if norm_name in _metadata_cache:
        # print(f"[spotify] Cache hit: {norm_name}")
        return _metadata_cache[norm_name]

    headers = _auth_headers()
    if not headers:
        return None

    # Clean up for better search results
    clean_name = norm_name # Use normalized name for search
    
    # Check if the filename follows "Artist - Title" format
    artist_query = ""
    track_query = ""
    if " - " in clean_name:
        parts = clean_name.split(" - ", 1)
        artist_query = parts[0].strip()
        track_query = parts[1].strip()
    elif "-" in clean_name:
        parts = clean_name.split("-", 1)
        artist_query = parts[0].strip()
        track_query = parts[1].strip()
    else:
        track_query = clean_name

    # Formulate a Spotify-specific advanced query
    if artist_query:
        sp_query = f"track:{track_query} artist:{artist_query}"
    else:
        sp_query = track_query

    print(f"[spotify] Searching: '{sp_query}'")

    try:
        response = requests.get(
            SPOTIFY_SEARCH_URL,
            headers=headers,
            params={"q": sp_query, "type": "track", "limit": 1, "market": "IN"},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        tracks = data.get("tracks", {}).get("items", [])
        
        if not tracks and artist_query:
            basic_query = f"{artist_query} {track_query}"
            print(f"[spotify] Fallback search: '{basic_query}'")
            response = requests.get(
                SPOTIFY_SEARCH_URL,
                headers=headers,
                params={"q": basic_query, "type": "track", "limit": 5, "market": "IN"},
                timeout=10
            )
            data = response.json()
            tracks = data.get("tracks", {}).get("items", [])

        if not tracks:
            print(f"[spotify] No results for: {clean_name}")
            return None

        # --- Fuzzy Matching Logic ---
        best_track = tracks[0]
        if len(tracks) > 1:
            highest_ratio = 0
            for t in tracks:
                t_name = t.get("name", "").lower()
                t_artist = t.get("artists", [{}])[0].get("name", "").lower()
                spotify_str = f"{t_artist} {t_name}"
                target_str = f"{artist_query.lower()} {track_query.lower()}".strip()
                ratio = SequenceMatcher(None, target_str, spotify_str).ratio()
                if ratio > highest_ratio:
                    highest_ratio = ratio
                    best_track = t
            print(f"[spotify] Fuzzy match ratio: {highest_ratio:.2f} for '{best_track.get('name')}'")

        track = best_track
        album = track.get("album", {})
        artists = track.get("artists", [])
        images = album.get("images", [])

        res = {
            "title":        track.get("name", ""),
            "artist":       artists[0].get("name", "") if artists else "",
            "artist_id":    artists[0].get("id", "") if artists else "",
            "track_id":     track.get("id", ""),
            "album":        album.get("name", ""),
            "release_date": album.get("release_date", ""),
            "image":        images[0].get("url", "") if images else "",
            "preview_url":  track.get("preview_url", ""),
            "spotify_url":  track.get("external_urls", {}).get("spotify", "")
        }
        
        # Save to cache
        _metadata_cache[norm_name] = res
        save_spotify_cache()
        return res

    except requests.RequestException as e:
        print(f"[spotify] Search error: {e}")
        return None


def get_artist_top_tracks(artist_id, market="IN", limit=5):
    """
    Fetch top tracks of an artist from Spotify.

    Returns list of dicts: [{ title, artist, album, preview_url, image }, ...]
    """
    if not artist_id:
        return []

    headers = _auth_headers()
    if not headers:
        return []

    try:
        url = f"{SPOTIFY_ARTISTS_URL}/{artist_id}/top-tracks"
        response = requests.get(
            url, headers=headers,
            params={"market": market},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        tracks = data.get("tracks", [])[:limit]
        results = []
        for t in tracks:
            album = t.get("album", {})
            images = album.get("images", [])
            results.append({
                "title":       t.get("name", ""),
                "artist":      t.get("artists", [{}])[0].get("name", ""),
                "album":       album.get("name", ""),
                "preview_url": t.get("preview_url", ""),
                "image":       images[0].get("url", "") if images else "",
                "spotify_url": t.get("external_urls", {}).get("spotify", "")
            })
        return results

    except requests.RequestException as e:
        print(f"[spotify] Artist top tracks error: {e}")
        return []


def get_track_details(track_id):
    """
    Fetch extended details for a specific track by its Spotify ID.

    Returns dict with: title, artist, all_artists, album, release_date,
    duration_ms, popularity, track_id, image, spotify_url, preview_url.
    Returns None on failure.
    """
    if not track_id:
        return None

    headers = _auth_headers()
    if not headers:
        return None

    try:
        url = f"https://api.spotify.com/v1/tracks/{track_id}"
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        track = response.json()

        album = track.get("album", {})
        artists = track.get("artists", [])
        images = album.get("images", [])

        return {
            "title":        track.get("name", ""),
            "artist":       artists[0].get("name", "") if artists else "",
            "all_artists":  ", ".join(a.get("name", "") for a in artists),
            "artist_id":    artists[0].get("id", "") if artists else "",
            "track_id":     track.get("id", ""),
            "album":        album.get("name", ""),
            "album_type":   album.get("album_type", ""),
            "release_date": album.get("release_date", ""),
            "duration_ms":  track.get("duration_ms", 0),
            "popularity":   track.get("popularity", 0),
            "explicit":     track.get("explicit", False),
            "disc_number":  track.get("disc_number", 1),
            "track_number": track.get("track_number", 1),
            "image":        images[0].get("url", "") if images else "",
            "preview_url":  track.get("preview_url", ""),
            "spotify_url":  track.get("external_urls", {}).get("spotify", "")
        }

    except requests.RequestException as e:
        print(f"[spotify] Track details error: {e}")
        return None


def get_similar_tracks(track_id, limit=5):
    """
    Fetch similar/recommended tracks using Spotify's recommendations API.

    Returns list of dicts: [{ title, artist, album, image, spotify_url }, ...]
    """
    if not track_id:
        return []

    headers = _auth_headers()
    if not headers:
        return []

    try:
        url = "https://api.spotify.com/v1/recommendations"
        response = requests.get(
            url, headers=headers,
            params={"seed_tracks": track_id, "limit": limit, "market": "IN"},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for t in data.get("tracks", []):
            album = t.get("album", {})
            images = album.get("images", [])
            artists = t.get("artists", [])
            results.append({
                "title":       t.get("name", ""),
                "artist":      artists[0].get("name", "") if artists else "",
                "album":       album.get("name", ""),
                "image":       images[0].get("url", "") if images else "",
                "spotify_url": t.get("external_urls", {}).get("spotify", "")
            })
        return results

    except requests.RequestException as e:
        print(f"[spotify] Recommendations error: {e}")
        return []


def get_youtube_search_url(title, artist=""):
    """
    Generate a YouTube search URL for a song.
    No API key required — just constructs the search URL.
    """
    query = f"{title} {artist}".strip()
    return f"https://www.youtube.com/results?search_query={quote_plus(query)}"
