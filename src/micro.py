import sounddevice as sd
import numpy as np
import wave
from pathlib import Path
from datetime import datetime


class Micro:
    def __init__(self, frequency: int = 16000, channels: int = 1, device=None):
        self.frequency = frequency
        self.channels = channels
        self.device = device
        self.is_recording = False
        self.audio_data = []
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[AUDIO STATUS] {status}")
        self.audio_data.append(indata.copy())

    def start_recording(self):
        if self.is_recording:
            print("Recording is already in progress.")
            return

        self.audio_data = []
        self.stream = sd.InputStream(
            samplerate=self.frequency,
            channels=self.channels,
            dtype="int16",
            device=self.device,
            callback=self._callback
        )
        self.stream.start()
        self.is_recording = True
        print("Recording started.")

    def stop_recording(self):
        if not self.is_recording:
            print("No recording in progress.")
            return

        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if self.audio_data:
            self.audio_data = np.concatenate(self.audio_data, axis=0)
        else:
            self.audio_data = np.empty((0, self.channels), dtype=np.int16)

        self.is_recording = False
        print("Recording stopped.")

    def record(self, duration: float):
        print(f"Recording for {duration} second(s)...")
        self.start_recording()
        sd.sleep(int(duration * 1000))
        self.stop_recording()

    def save_recording(self, filename: str | Path):
        if self.audio_data is None or len(self.audio_data) == 0:
            raise ValueError("No recording available to save.")

        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)

        with wave.open(str(filename), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.frequency)
            wf.writeframes(self.audio_data.tobytes())

        print(f"Saved: {filename}")

    def record_to_file(self, filename: str | Path, duration: float):
        self.record(duration)
        self.save_recording(filename)


def create_session_folder(base_dir: str | Path = "recordings") -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = Path(base_dir) / f"session_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def save_prompts_file(prompts: list[str], session_dir: str | Path, filename: str = "prompts.txt"):
    session_dir = Path(session_dir)
    prompt_path = session_dir / filename

    with open(prompt_path, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts, start=1):
            f.write(f"{i:02d}|{prompt}\n")

    print(f"Saved prompts: {prompt_path}")


def record_dataset(
    prompts: list[str],
    duration: float = 5.0,
    frequency: int = 16000,
    session_dir: str | Path | None = None,
    device=None,
    countdown: bool = True
) -> Path:
    if not prompts:
        raise ValueError("The prompts list is empty.")

    if session_dir is None:
        session_dir = create_session_folder()
    else:
        session_dir = Path(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)

    mic = Micro(frequency=frequency, channels=1, device=device)
    save_prompts_file(prompts, session_dir)

    total = len(prompts)
    print(f"\nStarting dataset recording: {total} phrase(s)")
    print(f"Output folder: {session_dir}\n")

    for i, prompt in enumerate(prompts, start=1):
        filename = session_dir / f"phrase_{i:02d}.wav"

        while True:
            print("=" * 60)
            print(f"Phrase {i}/{total}")
            print(prompt)
            input("Press Enter when ready...")

            if countdown:
                for t in [3, 2, 1]:
                    print(f"Recording in {t}...")
                    sd.sleep(1000)

            mic.record_to_file(filename, duration=duration)

            choice = input("Keep this recording? (y = yes / r = redo / q = quit): ").strip().lower()

            if choice == "y":
                print(f"Accepted -> {filename}\n")
                break
            elif choice == "r":
                print("Redoing this phrase...\n")
                continue
            elif choice == "q":
                print("Session stopped by user.")
                return Path(session_dir)
            else:
                print("Invalid choice. Keeping the recording by default.\n")
                break

    print("Dataset recording complete.")
    return Path(session_dir)


def transcribe_for(duration: float, filename: str = "recording.wav", frequency: int = 16000) -> str:
    mic = Micro(frequency=frequency)
    mic.record_to_file(filename, duration)
    return filename


def transcribe_directly(duration: float = 5.0, filename: str = "recording.wav", frequency: int = 16000) -> str:
    mic = Micro(frequency=frequency)
    mic.record_to_file(filename, duration)
    return filename


if __name__ == "__main__":
    phrases = [
        "Bonjour, comment allez-vous ?",
        "Je voudrais un verre d'eau.",
        "Le robot écoute attentivement.",
    ]

    record_dataset(
        prompts=phrases,
        duration=4,
        frequency=16000
    )