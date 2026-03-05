import os
import librosa
import soundfile as sf

# ─── Resolve the project and song directory paths absolutely ─────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))      # QBH_Project/
PARENT_DIR  = os.path.dirname(PROJECT_DIR)                    # qbh/

DEFAULT_INPUT_DIR  = os.path.join(PARENT_DIR, "songs")         # qbh/songs/
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_DIR, "wav_songs")    # QBH_Project/wav_songs/


def preprocess_audio(input_file, output_file, target_sr=22050):
    try:
        y, sr = librosa.load(input_file, sr=target_sr, mono=True)
        y = librosa.util.normalize(y)
        sf.write(output_file, y, target_sr)
        print(f"  OK  {os.path.basename(input_file)}  →  {os.path.basename(output_file)}")
        return True
    except Exception as e:
        print(f"  ERR {os.path.basename(input_file)}: {e}")
        return False


def convert_all_songs(input_dir=DEFAULT_INPUT_DIR, output_dir=DEFAULT_OUTPUT_DIR, target_sr=22050):
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_dir):
        print(f"Error: Songs directory '{input_dir}' does not exist.")
        return

    files = [f for f in os.listdir(input_dir)
             if f.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg', '.flac'))]

    if not files:
        print(f"No audio files found in '{input_dir}'.")
        return

    print(f"Found {len(files)} files in '{input_dir}'. Converting…")
    ok = 0
    for fn in files:
        inp  = os.path.join(input_dir, fn)
        base = os.path.splitext(fn)[0]
        out  = os.path.join(output_dir, base + ".wav")
        if preprocess_audio(inp, out, target_sr):
            ok += 1

    print(f"\nDone: {ok}/{len(files)} files converted → '{output_dir}'")


if __name__ == "__main__":
    print("Starting audio preprocessing…")
    convert_all_songs()
