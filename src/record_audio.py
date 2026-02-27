import sounddevice as sd
from scipy.io.wavfile import write
import time
import os
#from util import SENTENCES

# --- Configuration ---
FS = 16000  # Sample rate (standard for high quality)
DURATION = 10  # Seconds per recording
OUTPUT_DIR = "dataset_audio"

SENTENCES = [
    "Je mange du pain",
]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def record_session():
    print(f"--- Audio Collection Started ---")
    print(f"Goal: {len(SENTENCES)} sentences (~{len(SENTENCES) * DURATION / 60:.1f} minutes total)\n")
    separator = "-" * 30
    countdown_start = 3
    
    for i, sentence in enumerate(SENTENCES):
        print(f"[{i+1}/{len(SENTENCES)}] PLEASE READ CLEARLY:")
        print(f"\n>>> {sentence} <<<\n")
        
        for j in range(countdown_start, 0, -1):
            print(f"Starting in {j}...", end="\r")
            time.sleep(1)
        
        print("RECORDING...          ")
        
        audio_data = sd.rec(int(DURATION * FS), samplerate=FS, channels=1)
        sd.wait()
        
        filename = os.path.join(OUTPUT_DIR, f"sentence_{i+1:03d}.wav")
        write(filename, FS, audio_data)
        
        print(f"Saved: {filename}")
        print(separator)
        
if __name__ == "__main__":
    try:
        record_session()
    except KeyboardInterrupt:
        print("\nSession paused. You can resume later.")