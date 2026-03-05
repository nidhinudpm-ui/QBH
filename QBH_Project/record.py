import sounddevice as sd
import soundfile as sf
import os
import librosa
from convert import preprocess_audio

def record_audio(duration=10, filename="query.wav", sr=22050):
    print(f"Recording for {duration} seconds... Please start humming!")
    
    # Record audio to a NumPy array
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    
    # Wait until recording is finished
    sd.wait()
    print("Recording complete!")
    
    # Save the recorded audio as a temporary WAV file
    temp_filename = "temp_query.wav"
    sf.write(temp_filename, recording, sr)
    
    # Apply the same preprocessing as the database songs
    # This also normalizes the amplitude
    success = preprocess_audio(temp_filename, filename, target_sr=sr)
    
    if success:
        print(f"Humming saved and preprocessed successfully as '{filename}'.")
        # remove temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return True
    else:
        print("Failed to preprocess the recorded query.")
        return False

if __name__ == "__main__":
    record_audio()
