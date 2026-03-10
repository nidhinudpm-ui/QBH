import os
import sys
import base64
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from pydub import AudioSegment

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dejavu import Dejavu
from dejavu.recognize import FileRecognizer
from config import DEJAVU_DB_CONFIG, FFMPEG_PATH

# Ensure pydub uses our FFmpeg
AudioSegment.converter = FFMPEG_PATH

class DejavuService:
    def __init__(self):
        self.djv = Dejavu(DEJAVU_DB_CONFIG)

    def identify_from_file(self, file_path):
        """
        Identify a song from a given audio file path.
        """
        if not os.path.exists(file_path):
            return {"status": "error", "message": f"File not found: {file_path}"}
            
        try:
            # We use FileRecognizer for simple file/clip recognition
            results = self.djv.recognize(FileRecognizer, file_path)
            
            if results:
                # Results contains: song_id, song_name, confidence, offset, offset_seconds, file_sha1
                # The 'match_time' is added by recognize_file
                return {
                    "status": "success",
                    "match": results
                }
            else:
                return {
                    "status": "no_match",
                    "message": "No audio fingerprint match found in library."
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def generate_spectrogram(self, file_path):
        """
        Generate a base64 encoded PNG spectrogram of the audio file.
        """
        try:
            from dejavu import decoder
            from dejavu.fingerprint import DEFAULT_FS, DEFAULT_WINDOW_SIZE, DEFAULT_OVERLAP_RATIO
            
            # Limit to 45 seconds for speed
            channels, fs, _ = decoder.read(file_path, limit=45)
            data = channels[0] # Use first channel
            
            # Create figure
            plt.figure(figsize=(10, 4))
            plt.specgram(data, NFFT=DEFAULT_WINDOW_SIZE, Fs=fs, 
                        window=mlab.window_hanning, 
                        noverlap=int(DEFAULT_WINDOW_SIZE * DEFAULT_OVERLAP_RATIO))
            
            plt.axis('off')
            plt.tight_layout(pad=0)
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close()
            buf.seek(0)
            
            return base64.b64encode(buf.read()).decode('utf-8')
        except Exception as e:
            print(f"[DejavuService] Spectrogram error: {e}")
            return None

# Singleton instance
_service = None

def get_dejavu_service():
    global _service
    if _service is None:
        _service = DejavuService()
    return _service

if __name__ == "__main__":
    # Test block
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        svc = get_dejavu_service()
        print(f"Testing recognition for: {test_file}")
        print(svc.identify_from_file(test_file))
    else:
        print("Usage: python dejavu_service.py <audio_file_path>")
