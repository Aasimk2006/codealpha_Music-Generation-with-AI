"""
Data Collector Module for AI Music Generation
Collects MIDI music from music21's built-in corpus (Classical & Ragtime/Early Jazz),
generates structured Jazz/Blues progression MIDI files, and organizes files into genres.
"""

import os
from pathlib import Path
from typing import List, Optional
import music21


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MIDI_DIR = DATA_DIR / "midi"
CLASSICAL_DIR = MIDI_DIR / "classical"
JAZZ_DIR = MIDI_DIR / "jazz"


def ensure_directories():
    """Ensure all required directories exist."""
    CLASSICAL_DIR.mkdir(parents=True, exist_ok=True)
    JAZZ_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)


def extract_classical_corpus(max_pieces: int = 25) -> List[Path]:
    """
    Extract classical pieces (Bach, Beethoven, Mozart, Haydn) from music21 corpus
    and export them to standard MIDI files.
    """
    ensure_directories()
    exported_files = []
    
    composers = [
        ("bach", min(15, max_pieces)),
        ("beethoven", min(5, max(1, max_pieces // 4))),
        ("mozart", min(5, max(1, max_pieces // 4))),
    ]
    
    total_saved = 0
    for composer, count in composers:
        if total_saved >= max_pieces:
            break
        try:
            works = music21.corpus.getComposer(composer)
            for i, work_path in enumerate(works[:count]):
                if total_saved >= max_pieces:
                    break
                out_path = CLASSICAL_DIR / f"{composer}_piece_{i+1:02d}.mid"
                if not out_path.exists():
                    try:
                        parsed = music21.corpus.parse(work_path)
                        parsed.write("midi", fp=str(out_path))
                        exported_files.append(out_path)
                        total_saved += 1
                        print(f"[Classical] Exported {out_path.name}")
                    except Exception as err:
                        print(f"[Classical] Warning: Skipping {work_path}: {err}")
                else:
                    exported_files.append(out_path)
                    total_saved += 1
        except Exception as e:
            print(f"[Classical] Could not load {composer}: {e}")
            
    return exported_files


def generate_jazz_and_blues_midi(num_variations: int = 15) -> List[Path]:
    """
    Generate authentic jazz chord progressions (ii-V-I, 12-bar blues, bebop lines,
    swing rhythms, and modal jazz) and export to MIDI. Also includes Scott Joplin ragtime.
    """
    ensure_directories()
    exported_files = []

    # 1. Include Scott Joplin ragtime from music21 corpus if available
    try:
        joplin_works = music21.corpus.getComposer("joplin")
        for i, jop_work in enumerate(joplin_works):
            out_path = JAZZ_DIR / f"joplin_ragtime_{i+1:02d}.mid"
            if not out_path.exists():
                try:
                    parsed = music21.corpus.parse(jop_work)
                    parsed.write("midi", fp=str(out_path))
                    exported_files.append(out_path)
                    print(f"[Jazz] Exported {out_path.name}")
                except Exception as ex:
                    print(f"[Jazz] Warning: Skipping Joplin piece: {ex}")
            else:
                exported_files.append(out_path)
    except Exception as e:
        print(f"[Jazz] Joplin retrieval info: {e}")

    # 2. Procedural Jazz Progressions (ii-V-I, Autumn Leaves, 12-Bar Blues, Modal Dorian)
    jazz_styles = [
        # (Name, Chords, Scales/Melody notes, Tempo)
        (
            "ii_V_I_swing",
            [
                ("D4.F4.A4.C5", 2.0), ("G3.B3.D4.F4", 2.0), ("C4.E4.G4.B4", 4.0),
                ("E4.G4.B4.D5", 2.0), ("A3.C#4.E4.G4", 2.0), ("D4.F#4.A4.C#5", 4.0)
            ],
            ["D4", "F4", "A4", "C5", "B4", "G4", "F4", "E4", "G4", "C5", "B4", "C5"],
            120
        ),
        (
            "12_bar_blues_in_F",
            [
                ("F3.A3.C4.Eb4", 4.0), ("Bb3.D4.F4.Ab4", 4.0), ("F3.A3.C4.Eb4", 4.0), ("F3.A3.C4.Eb4", 4.0),
                ("Bb3.D4.F4.Ab4", 4.0), ("Bb3.D4.F4.Ab4", 4.0), ("F3.A3.C4.Eb4", 4.0), ("F3.A3.C4.Eb4", 4.0),
                ("C4.E4.G4.Bb4", 4.0), ("Bb3.D4.F4.Ab4", 4.0), ("F3.A3.C4.Eb4", 4.0), ("C4.E4.G4.Bb4", 4.0)
            ],
            ["F4", "Ab4", "A4", "C5", "D5", "Eb5", "D5", "C5", "Ab4", "F4"],
            110
        ),
        (
            "modal_dorian_cool_jazz",
            [
                ("D3.A3.C4.F4", 4.0), ("Eb3.Bb3.Db4.Gb4", 4.0),
                ("D3.A3.C4.F4", 4.0), ("G3.D4.F4.Bb4", 4.0)
            ],
            ["D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "C5", "A4", "F4", "D4"],
            100
        ),
        (
            "bebop_rhythm_changes",
            [
                ("Bb3.D4.F4.A4", 2.0), ("G3.B3.D4.F4", 2.0), ("C4.Eb4.G4.Bb4", 2.0), ("F3.A3.C4.Eb4", 2.0),
                ("D4.F4.Ab4.C5", 2.0), ("G3.B3.D4.F4", 2.0), ("C4.E4.G4.Bb4", 2.0), ("F3.A3.C4.Eb4", 2.0)
            ],
            ["Bb4", "C5", "D5", "F5", "D5", "Eb5", "E5", "F5", "G5", "F5", "D5", "Bb4"],
            140
        )
    ]

    for style_idx in range(num_variations):
        style_name, chord_seq, mel_notes, tempo_bpm = jazz_styles[style_idx % len(jazz_styles)]
        out_path = JAZZ_DIR / f"jazz_{style_name}_var{style_idx+1:02d}.mid"
        if out_path.exists():
            exported_files.append(out_path)
            continue

        score = music21.stream.Score()
        part = music21.stream.Part()
        part.append(music21.tempo.MetronomeMark(number=tempo_bpm))

        # Interleave chords with melodic swing runs
        for repeat in range(4):
            for chord_str, duration_val in chord_seq:
                # Add chord
                pitch_names = chord_str.split(".")
                c = music21.chord.Chord(pitch_names)
                c.duration = music21.duration.Duration(duration_val)
                part.append(c)

                # Add melodic syncopated swing embellishments
                for m_idx in range(len(mel_notes) // 3):
                    mel_note_name = mel_notes[(repeat + m_idx) % len(mel_notes)]
                    n = music21.note.Note(mel_note_name)
                    # Swing rhythm variation: dotted eighth + sixteenth
                    n.duration = music21.duration.Duration(0.5 if (m_idx % 2 == 0) else 0.25)
                    part.append(n)

        score.append(part)
        try:
            score.write("midi", fp=str(out_path))
            exported_files.append(out_path)
            print(f"[Jazz] Created {out_path.name}")
        except Exception as err:
            print(f"[Jazz] Error writing {out_path.name}: {err}")

    return exported_files


def collect_all_data(classical_count: int = 20, jazz_count: int = 15) -> dict:
    """
    Main entry point to collect both classical and jazz datasets.
    """
    print(f"--- Collecting MIDI Data (Classical & Jazz) ---")
    classical_files = extract_classical_corpus(max_pieces=classical_count)
    jazz_files = generate_jazz_and_blues_midi(num_variations=jazz_count)
    
    print(f"Data collection completed:")
    print(f" - Classical MIDI files: {len(classical_files)} at {CLASSICAL_DIR}")
    print(f" - Jazz MIDI files:      {len(jazz_files)} at {JAZZ_DIR}")
    
    return {
        "classical": classical_files,
        "jazz": jazz_files
    }


if __name__ == "__main__":
    collect_all_data()
