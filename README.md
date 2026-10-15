# 🎵 AI Music Generation with Deep Learning (LSTM) & music21

An end-to-end Deep Learning system that learns musical patterns from classical and jazz MIDI data, trains an LSTM/GRU recurrent neural network, autoregressively generates novel melodies and harmonies, and synthesizes them into standard MIDI (`.mid`) and audio (`.wav`) files.

---

## 🌟 Key Features

1. **MIDI Music Data Collection**:
   - Curated classical repertoire (Bach, Beethoven, Mozart) from `music21.corpus`.
   - Ragtime and authentic jazz chord progressions (Scott Joplin, ii-V-I swing, 12-bar blues in F, modal Dorian cool jazz, bebop rhythm changes).
   - Easily drop custom `.mid` / `.midi` files into `data/midi/classical` or `data/midi/jazz`.

2. **music21 Preprocessing**:
   - Parses polyphonic MIDI streams into pitch sequences, multi-note chords (e.g., `C4.E4.G4`), and rhythmic rests.
   - Builds integer vocabulary mappings and sliding-window training pairs ($X_{32} \to y_{next}$).
   - Preprocessed dataset: **75,308 sequence pairs** across **316 unique musical tokens**.

3. **Deep Neural Network Architectures**:
   - **`MusicLSTM`**: Multi-layer LSTM with musical pitch embeddings, dropout regularization, and dense projection layers.
   - **`MusicGRU`**: Efficient gated recurrent alternative for fast training.
   - **`MusicGAN`**: Generator and Discriminator adversarial network architecture.

4. **Autoregressive Generation**:
   - Temperature-controlled sampling ($0.5$ for structured classical, $1.2$ for exploratory jazz).
   - Top-K candidate filtering to prune unnatural discordant intervals.

5. **Audio Synthesis & MIDI Export**:
   - Outputs standard multi-track MIDI (`.mid`) files.
   - Native polyphonic acoustic harmonic waveform synthesizer that renders playable `.wav` audio with ADSR envelopes (no external soundfont or binary dependencies required).
   - In-app and terminal speaker playback via `pygame.mixer`.

6. **Interactive Web Dashboard & CLI**:
   - **Streamlit Web Studio** (`app.py`): Visual music generation, in-browser audio playback, note contour plots, and live training progress.
   - **Unified CLI** (`main.py`): Full headless command-line interface.

---

## 📁 Repository Structure

```
Music Generation with AI/
├── app.py                  # Interactive Streamlit Web UI
├── main.py                 # Unified CLI application
├── requirements.txt        # Python dependencies
├── README.md               # Documentation & Guide
│
├── data/
│   ├── midi/               # Collected MIDI datasets
│   │   ├── classical/      # Bach, Beethoven, Mozart pieces
│   │   └── jazz/           # Joplin ragtime, ii-V-I swing, blues
│   └── processed/          # Preprocessed data & vocabulary
│       ├── X.npy           # Input sequence training tensors
│       ├── y.npy           # Target note training tensors
│       ├── vocab.json      # Token <-> Integer mappings
│       ├── metadata.json   # Dataset metrics and parameters
│       └── notes.pkl       # Parsed token sequence
│
├── checkpoints/            # Trained neural network weights
│   ├── best_music_lstm.pt  # Optimal LSTM weights
│   └── history.json        # Loss metrics across epochs
│
├── output/                 # Generated musical compositions
│   ├── midi/               # Output .mid files
│   └── audio/              # Synthesized .wav audio files
│
└── src/
    ├── __init__.py
    ├── data_collector.py   # MIDI data extraction & generation
    ├── preprocess.py       # music21 parsing & tokenization
    ├── dataset.py          # PyTorch Dataset & DataLoader
    ├── model.py            # Deep LSTM, GRU & GAN architectures
    ├── train.py            # Training pipeline with AdamW & scheduler
    ├── generate.py         # Autoregressive sequence generator
    ├── audio_synth.py      # Polyphonic harmonic WAV synthesis & MIDI export
    └── playback.py         # Pygame audio player
```

---

## 🚀 Quick Start Guide

### 1. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Launch the Web Studio (Recommended)
Run the interactive Streamlit dashboard:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser to compose music, listen directly to generated audio tracks, and download `.mid` and `.wav` files.

---

## 💻 CLI Usage

You can run individual stages or the entire pipeline using `main.py`:

### Generate a New Song (Instant):
```bash
python main.py --generate --notes 80 --temp 0.85 --name my_song --play
```

### Run the Full Pipeline from Scratch:
```bash
python main.py --pipeline --epochs 8 --notes 100 --name master_composition
```

### Individual Commands:
- **Collect MIDI Data**:
  ```bash
  python main.py --collect
  ```
- **Preprocess Dataset**:
  ```bash
  python main.py --preprocess
  ```
- **Train LSTM Model**:
  ```bash
  python main.py --train --epochs 10 --model lstm
  ```
- **Play Latest Generated Audio**:
  ```bash
  python main.py --play --name my_song
  ```

---

## 📊 Model Training Performance

The model was trained on **75,308 sequence pairs** with a vocabulary of **316 unique notes and chords**:

| Epoch | Train Loss | Validation Loss |
| :---: | :---: | :---: |
| 1 | 3.9754 | 3.6515 |
| 2 | 3.4684 | 3.2794 |
| 3 | 3.1523 | 3.0124 |
| 4 | 2.9156 | 2.8165 |
| 5 | 2.7028 | 2.6277 |
| 6 | 2.5067 | 2.4457 |
| 7 | 2.3273 | 2.3202 |
| **8** | **2.1641** | **2.1717** |

---

## 🛠️ Adding Your Own Music
To train the model on your own favorite MIDI songs:
1. Place any `.mid` or `.midi` files into `data/midi/classical/` or `data/midi/jazz/`.
2. Run `python main.py --preprocess`.
3. Train the model with `python main.py --train --epochs 10`.
4. Generate new songs with your personalized style!

---

## 👨‍💻 Developed By

- **Developer:** **AASIM KHAN**
- **Candidate ID:** `CA/DF1/312332`

