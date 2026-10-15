"""
Playback Module
Plays synthesized WAV audio or MIDI files through system speakers using pygame.mixer.
Supports interactive playback with duration limits.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pygame


def play_audio(file_path: Union[str, Path] if False else Path, duration_sec: Optional[float] = None):
    """
    Plays audio or MIDI file through pygame.mixer.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return

    print(f"\n[Playing] {file_path.name} ...")
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(str(file_path))
        pygame.mixer.music.play()

        start = time.time()
        while pygame.mixer.music.get_busy():
            if duration_sec and (time.time() - start) >= duration_sec:
                print(f"[Playback] Reached duration limit ({duration_sec}s). Stopping.")
                pygame.mixer.music.stop()
                break
            time.sleep(0.1)

        print("[Playback] Finished.")
    except Exception as e:
        print(f"Playback error: {e}")
    finally:
        try:
            pygame.mixer.quit()
        except Exception:
            pass


def stop_audio():
    """Stops any currently playing audio."""
    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.quit()
    except Exception:
        pass


if __name__ == "__main__":
    demo_wav = Path(__file__).resolve().parent.parent / "output" / "audio" / "demo_test.wav"
    if demo_wav.exists():
        play_audio(demo_wav, duration_sec=2.0)
