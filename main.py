"""
Main CLI Application for AI Music Generation
Unified command-line interface for:
- Data Collection
- music21 Preprocessing
- Deep Learning (LSTM / GRU) Training
- AI Music Sequence Generation
- MIDI & WAV Audio Conversion and Playback
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collector import collect_all_data
from src.preprocess import preprocess_dataset
from src.train import train_model
from src.generate import generate_music_sequence
from src.audio_synth import convert_and_save_all
from src.playback import play_audio


def run_pipeline(
    genre: str = "all",
    epochs: int = 8,
    num_notes: int = 100,
    temperature: float = 0.85,
    song_name: str = "ai_music_composition"
):
    """
    Executes the complete end-to-end AI Music Generation pipeline.
    """
    print("=" * 65)
    print("      AI MUSIC GENERATION PIPELINE (END-TO-END)")
    print("=" * 65)

    # 1. Collect Data
    print("\n[STEP 1/5] Collecting MIDI Music Dataset...")
    collect_all_data()

    # 2. Preprocess Data
    print("\n[STEP 2/5] Preprocessing into Note Sequences with music21...")
    preprocess_dataset(genre=None if genre == "all" else genre, sequence_length=32)

    # 3. Train Model
    print(f"\n[STEP 3/5] Training Deep LSTM Model for {epochs} Epochs...")
    train_model(model_type="lstm", epochs=epochs, batch_size=256, learning_rate=0.002)

    # 4. Generate Music Sequence
    print(f"\n[STEP 4/5] Generating {num_notes} Note Musical Sequence (Temp: {temperature})...")
    notes = generate_music_sequence(num_notes=num_notes, temperature=temperature)

    # 5. Convert to MIDI and WAV Audio
    print("\n[STEP 5/5] Converting to MIDI and Synthesizing WAV Audio...")
    files = convert_and_save_all(notes, base_name=song_name)

    print("\n" + "=" * 65)
    print("Pipeline Finished Successfully!")
    print(f"Generated MIDI File:  {files['midi_path']}")
    print(f"Generated Audio File: {files['wav_path']}")
    print("=" * 65)
    return files


def main():
    parser = argparse.ArgumentParser(
        description="AI Music Generation with Deep Learning (LSTM) & music21"
    )
    parser.add_argument("--collect", action="store_true", help="Collect MIDI music data (classical & jazz)")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess MIDI dataset into note sequences")
    parser.add_argument("--train", action="store_true", help="Train the deep learning model")
    parser.add_argument("--generate", action="store_true", help="Generate a new music sequence")
    parser.add_argument("--play", action="store_true", help="Play the latest generated audio")
    parser.add_argument("--pipeline", action="store_true", help="Run full end-to-end pipeline")

    # Options
    parser.add_argument("--genre", type=str, default="all", choices=["all", "classical", "jazz"], help="Genre focus")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "gru"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--notes", type=int, default=100, help="Number of notes to generate")
    parser.add_argument("--temp", type=float, default=0.85, help="Generation temperature (0.5=formal, 1.2=creative)")
    parser.add_argument("--name", type=str, default="generated_piece", help="Output base filename")

    args = parser.parse_args()

    # If no flags passed, display help or run generation if model checkpoint exists
    if not any([args.collect, args.preprocess, args.train, args.generate, args.play, args.pipeline]):
        checkpoint = PROJECT_ROOT / "checkpoints" / "best_music_lstm.pt"
        if checkpoint.exists():
            print("Checkpoint found! Generating a new piece of music...")
            args.generate = True
        else:
            parser.print_help()
            sys.exit(0)

    if args.pipeline:
        run_pipeline(
            genre=args.genre,
            epochs=args.epochs,
            num_notes=args.notes,
            temperature=args.temp,
            song_name=args.name
        )
        return

    if args.collect:
        collect_all_data()

    if args.preprocess:
        preprocess_dataset(genre=None if args.genre == "all" else args.genre)

    if args.train:
        train_model(model_type=args.model, epochs=args.epochs, batch_size=256, learning_rate=0.002)

    if args.generate:
        print(f"\nGenerating {args.notes} musical events...")
        notes = generate_music_sequence(num_notes=args.notes, temperature=args.temp)
        files = convert_and_save_all(notes, base_name=args.name)
        print("\nGenerated Output:")
        print(f" - MIDI:  {files['midi_path']}")
        print(f" - Audio: {files['wav_path']}")

        if args.play:
            play_audio(files["wav_path"], duration_sec=15.0)

    elif args.play:
        latest_wav = PROJECT_ROOT / "output" / "audio" / f"{args.name}.wav"
        if not latest_wav.exists():
            # Try finding any wav in output/audio
            wavs = list((PROJECT_ROOT / "output" / "audio").glob("*.wav"))
            if wavs:
                latest_wav = wavs[-1]
            else:
                print("No audio file found to play. Run --generate first!")
                return
        play_audio(latest_wav, duration_sec=15.0)


if __name__ == "__main__":
    main()
