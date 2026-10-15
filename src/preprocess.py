"""
Music Preprocessing Module
Extracts note and chord sequences from MIDI files using music21,
builds token vocabularies, and prepares training sequences.
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import music21
import numpy as np


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MIDI_DIR = DATA_DIR / "midi"
PROCESSED_DIR = DATA_DIR / "processed"


def parse_midi_file(file_path: Path) -> List[str]:
    """
    Parses a single MIDI file into a sequence of note and chord tokens.
    """
    notes_sequence = []
    try:
        midi_data = music21.converter.parse(str(file_path))
        # Flatten parts and extract notes, chords, and rests
        elements = midi_data.flatten().notesAndRests
        
        for element in elements:
            if isinstance(element, music21.note.Note):
                # Pitch name with octave e.g. 'C4', 'F#3'
                notes_sequence.append(str(element.pitch))
            elif isinstance(element, music21.chord.Chord):
                # Sort chord pitches from lowest to highest for canonical representation
                pitches = sorted([str(p) for p in element.pitches])
                notes_sequence.append(".".join(pitches))
            elif isinstance(element, music21.note.Rest):
                # Optional: include rest if desired
                quarter_len = round(float(element.duration.quarterLength), 2)
                if quarter_len >= 0.5:
                    notes_sequence.append("REST")
                    
    except Exception as e:
        print(f"Warning: Could not parse {file_path.name}: {e}")
        
    return notes_sequence


def extract_notes_from_directory(
    directory_path: Path,
    genre: Optional[str] = None
) -> List[str]:
    """
    Extracts note tokens from all MIDI files in a directory or genre subdirectory.
    """
    all_notes = []
    if genre and (directory_path / genre).exists():
        search_dirs = [directory_path / genre]
    else:
        # Search all subdirectories (classical, jazz, etc.)
        subdirs = [d for d in directory_path.iterdir() if d.is_dir()]
        search_dirs = subdirs if subdirs else [directory_path]

    midi_files = []
    for d in search_dirs:
        for ext in ("*.mid", "*.midi", "*.MID", "*.MIDI"):
            midi_files.extend(d.glob(ext))

    print(f"Found {len(midi_files)} MIDI files across {[d.name for d in search_dirs]}.")
    
    for i, midi_file in enumerate(midi_files, 1):
        parsed = parse_midi_file(midi_file)
        if parsed:
            all_notes.extend(parsed)
            # Add a song boundary delimiter token if multiple songs
            all_notes.append("<SEP>")
        if i % 10 == 0 or i == len(midi_files):
            print(f"Processed {i}/{len(midi_files)} files. Total note tokens so far: {len(all_notes)}")

    return all_notes


def build_vocabulary(notes: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Builds bidirectional vocabulary mapping unique notes/chords to integers.
    """
    unique_tokens = sorted(list(set(notes)))
    note_to_int = {token: idx for idx, token in enumerate(unique_tokens)}
    int_to_note = {idx: token for idx, token in enumerate(unique_tokens)}
    return note_to_int, int_to_note


def prepare_sequences(
    notes: List[str],
    note_to_int: Dict[str, int],
    sequence_length: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepares input (X) and target (y) sequence pairs using a sliding window.
    X: sequences of length `sequence_length`
    y: the next note to predict
    """
    network_input = []
    network_output = []

    # Filter out boundary token during sliding window or let it act as sequence separator
    encoded_notes = [note_to_int[token] for token in notes]

    for i in range(0, len(encoded_notes) - sequence_length):
        seq_in = encoded_notes[i : i + sequence_length]
        seq_out = encoded_notes[i + sequence_length]
        network_input.append(seq_in)
        network_output.append(seq_out)

    X = np.array(network_input, dtype=np.int64)
    y = np.array(network_output, dtype=np.int64)
    return X, y


def preprocess_dataset(
    genre: Optional[str] = None,
    sequence_length: int = 32,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Full preprocessing pipeline:
    1. Extracts notes from MIDI files
    2. Builds vocabulary
    3. Prepares sliding window sequence pairs
    4. Saves files to data/processed/
    """
    out_dir = output_dir or PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Starting MIDI Preprocessing (Genre: {genre or 'All'}) ---")
    raw_notes = extract_notes_from_directory(MIDI_DIR, genre=genre)

    if not raw_notes:
        raise ValueError(f"No MIDI files found or could not parse any notes in {MIDI_DIR}")

    note_to_int, int_to_note = build_vocabulary(raw_notes)
    vocab_size = len(note_to_int)

    X, y = prepare_sequences(raw_notes, note_to_int, sequence_length=sequence_length)

    print(f"\nPreprocessing Complete:")
    print(f" - Total note events:     {len(raw_notes)}")
    print(f" - Unique vocabulary:     {vocab_size} tokens")
    print(f" - Total training pairs:  {len(X)} sequences (seq_len={sequence_length})")

    # Save artifacts
    notes_file = out_dir / "notes.pkl"
    with open(notes_file, "wb") as f:
        pickle.dump(raw_notes, f)

    vocab_file = out_dir / "vocab.json"
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump({
            "note_to_int": note_to_int,
            "int_to_note": {str(k): v for k, v in int_to_note.items()}
        }, f, indent=2)

    metadata = {
        "genre": genre or "all",
        "total_notes": len(raw_notes),
        "vocab_size": vocab_size,
        "sequence_length": sequence_length,
        "num_sequences": len(X),
        "classes": list(note_to_int.keys())
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save numpy dataset arrays
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)

    print(f"Saved preprocessed data to {out_dir}")
    return metadata


def load_preprocessed_data(processed_dir: Optional[Path] = None):
    """
    Loads preprocessed dataset and vocabulary from disk.
    """
    p_dir = processed_dir or PROCESSED_DIR
    with open(p_dir / "vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    note_to_int = vocab["note_to_int"]
    int_to_note = {int(k): v for k, v in vocab["int_to_note"].items()}

    with open(p_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    X = np.load(p_dir / "X.npy")
    y = np.load(p_dir / "y.npy")
    
    with open(p_dir / "notes.pkl", "rb") as f:
        raw_notes = pickle.load(f)

    return X, y, note_to_int, int_to_note, metadata, raw_notes


if __name__ == "__main__":
    preprocess_dataset(sequence_length=32)
