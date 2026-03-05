"""
spotify_client.py — Spotify Web API integration

Uses Client Credentials flow (no user login needed).
Credentials are loaded securely from .env via config.py.
Token is cached and reused until expiry.
"""

import time
import requests
from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET

# ─── Token Cache ──────────────────────────────────────────────────────────────
_token_cache = {"access_token": None, "expires_at": 0}

SPOTIFY_TOKEN_URL  = "https://accounts.spotify.com/api/token"
SPOTIFY_SEARCH_URL = "https://api.spotify.com/v1/search"
SPOTIFY_ARTISTS_URL = "https://api.spotify.com/v1/artists"


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

    Returns dict with: title, artist, artist_id, album, release_date, image, preview_url
    Returns None on failure.
    """
    headers = _auth_headers()
    if not headers:
        return None

    # Clean up the song name for better search results
    clean_name = song_name.replace(".wav", "").replace("_spotdown.org", "")
    clean_name = clean_name.replace("_", " ").replace("  ", " ").strip()

    # Remove common filename artifacts
    for pattern in ["(From ", "From _", "_ ", "_"]:
        clean_name = clean_name.replace(pattern, " ")
    clean_name = clean_name.replace("  ", " ").strip()

    try:
        response = requests.get(
            SPOTIFY_SEARCH_URL,
            headers=headers,
            params={"q": clean_name, "type": "track", "limit": 1, "market": "IN"},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        tracks = data.get("tracks", {}).get("items", [])
        if not tracks:
            print(f"[spotify] No results for: {clean_name}")
            return None

        track = tracks[0]
        album = track.get("album", {})
        artists = track.get("artists", [])
        images = album.get("images", [])

        return {
            "title":        track.get("name", ""),
            "artist":       artists[0].get("name", "") if artists else "",
            "artist_id":    artists[0].get("id", "") if artists else "",
            "album":        album.get("name", ""),
            "release_date": album.get("release_date", ""),
            "image":        images[0].get("url", "") if images else "",
            "preview_url":  track.get("preview_url", ""),
            "spotify_url":  track.get("external_urls", {}).get("spotify", "")
        }

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
