import os
import sys
from pydub import AudioSegment

# Add project root to sys.path for internal imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dejavu import Dejavu
from config import DEJAVU_DB_CONFIG, AUDIO_FINGERPRINT_SONGS_DIR, FFMPEG_PATH, FINGERPRINT_EXTENSIONS

# Configure pydub to use our imageio-ffmpeg binary
AudioSegment.converter = FFMPEG_PATH

def build_database():
    print("--- Dejavu Database Builder ---")
    print(f"Target Songs Directory: {AUDIO_FINGERPRINT_SONGS_DIR}")
    
    if not os.path.exists(AUDIO_FINGERPRINT_SONGS_DIR):
        print(f"Error: Songs directory not found at {AUDIO_FINGERPRINT_SONGS_DIR}")
        return

    # Initialize Dejavu
    djv = Dejavu(DEJAVU_DB_CONFIG)

    # Fingerprint all files in the directory
    # Dejavu handles subdirectories and extension filtering
    print("Starting fingerprinting process...")
    # remove leading dots from extensions for dejavu
    extensions = [e.replace(".", "") for e in FINGERPRINT_EXTENSIONS]
    
    djv.fingerprint_directory(AUDIO_FINGERPRINT_SONGS_DIR, extensions)
    
    print("\n--- Fingerprinting Complete ---")
    print(f"Songs in DB: {djv.db.get_num_songs()}")
    print(f"Fingerprints in DB: {djv.db.get_num_fingerprints()}")

if __name__ == "__main__":
    build_database()
