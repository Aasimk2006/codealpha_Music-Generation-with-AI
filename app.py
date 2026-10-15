"""
Streamlit Web Application: AI Music Generation Studio
Interactive web UI for exploring datasets, training deep neural networks,
generating AI musical compositions, playing synthesized audio, and downloading MIDI/WAV files.
"""

import sys
import json
from pathlib import Path
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collector import collect_all_data, CLASSICAL_DIR, JAZZ_DIR
from src.preprocess import preprocess_dataset, PROCESSED_DIR
from src.train import train_model, CHECKPOINT_DIR
from src.generate import generate_music_sequence
from src.audio_synth import convert_and_save_all, OUTPUT_MIDI_DIR, OUTPUT_AUDIO_DIR


# Page Setup
st.set_page_config(
    page_title="AI Music Generation Studio",
    page_icon="🎵",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #8B5CF6, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #6B7280;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎵 AI Music Generation Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Generate classical and jazz compositions using Deep Recurrent Neural Networks (LSTM) & music21</div>', unsafe_allow_html=True)

# Tabs
tab_generate, tab_explore, tab_train, tab_about = st.tabs([
    "✨ Generate Music",
    "🎼 Dataset & Vocabulary",
    "🧠 Train AI Model",
    "📖 How It Works"
])


# -----------------------------------------------------------------------------
# TAB 1: GENERATE MUSIC
# -----------------------------------------------------------------------------
with tab_generate:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("🎛️ Generation Controls")
        
        # Checkpoint check
        checkpoint_files = list(CHECKPOINT_DIR.glob("*.pt"))
        if not checkpoint_files:
            st.warning("No trained checkpoint found! Please train a model first in the 'Train AI Model' tab.")
            selected_ckpt = None
        else:
            selected_ckpt = st.selectbox(
                "Model Checkpoint",
                options=checkpoint_files,
                format_func=lambda x: x.name
            )

        num_notes = st.slider("Sequence Length (Notes/Chords)", min_value=20, max_value=200, value=70, step=10)
        
        temperature = st.slider(
            "Creativity / Temperature",
            min_value=0.5,
            max_value=1.4,
            value=0.85,
            step=0.05,
            help="Lower (0.5-0.7) = structured, classical, predictable. Higher (1.0-1.4) = expressive, jazz-like, experimental."
        )

        top_k = st.slider(
            "Top-K Sampling Filter",
            min_value=5,
            max_value=50,
            value=15,
            help="Limits selection to the K highest probability candidate notes at each step to prevent erratic notes."
        )

        composition_title = st.text_input("Composition Title", value="ai_harmony_session")
        tempo_bpm = st.slider("Tempo (BPM)", min_value=60, max_value=180, value=115, step=5)

        generate_btn = st.button("🎶 Generate AI Music", type="primary", use_container_width=True)

    with col_right:
        st.subheader("🎧 Generated Composition")

        if generate_btn and selected_ckpt:
            with st.spinner("Neural network is composing music sequence..."):
                try:
                    notes = generate_music_sequence(
                        num_notes=num_notes,
                        temperature=temperature,
                        top_k=top_k,
                        checkpoint_file=selected_ckpt
                    )
                    
                    st.session_state["latest_notes"] = notes

                    with st.spinner("Synthesizing polyphonic WAV audio and MIDI file..."):
                        files = convert_and_save_all(
                            notes,
                            base_name=composition_title,
                            tempo_bpm=tempo_bpm
                        )
                        st.session_state["latest_files"] = files
                    
                    st.success(f"Composition '{composition_title}' generated successfully!")
                except Exception as e:
                    st.error(f"Error during generation: {e}")

        # Display latest generated results
        if "latest_files" in st.session_state:
            files = st.session_state["latest_files"]
            wav_path = Path(files["wav_path"])
            midi_path = Path(files["midi_path"])

            if wav_path.exists():
                st.markdown("#### 🔊 Audio Player")
                with open(wav_path, "rb") as audio_file:
                    st.audio(audio_file.read(), format="audio/wav")

                # Download Buttons
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    with open(wav_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download Audio (.wav)",
                            data=f.read(),
                            file_name=wav_path.name,
                            mime="audio/wav",
                            use_container_width=True
                        )
                with btn_col2:
                    with open(midi_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download MIDI (.mid)",
                            data=f.read(),
                            file_name=midi_path.name,
                            mime="audio/midi",
                            use_container_width=True
                        )

            # Melody / Sequence Visualizer
            if "latest_notes" in st.session_state:
                notes = st.session_state["latest_notes"]
                st.markdown("#### 🎼 Note Sequence Preview")
                preview_str = "  →  ".join(notes[:25])
                st.code(preview_str + ("  →  ..." if len(notes) > 25 else ""), language="text")

                # Visual Plot of Note Events
                fig, ax = plt.subplots(figsize=(10, 3.2))
                pitches_numeric = []
                for n in notes:
                    if "." in n:
                        # Extract first pitch
                        n = n.split(".")[0]
                    if n == "REST":
                        pitches_numeric.append(np.nan)
                    else:
                        try:
                            import music21
                            pitches_numeric.append(music21.pitch.Pitch(n).midi)
                        except Exception:
                            pitches_numeric.append(60)

                ax.plot(pitches_numeric, marker="o", color="#8B5CF6", linewidth=1.5, markersize=4)
                ax.set_title("Melodic Contour (MIDI Pitch vs Time Step)", fontsize=11, fontweight="bold")
                ax.set_xlabel("Time Step (Note Index)")
                ax.set_ylabel("MIDI Pitch Number")
                ax.grid(True, linestyle="--", alpha=0.5)
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("Click '**Generate AI Music**' on the left to compose a new piece!")


# -----------------------------------------------------------------------------
# TAB 2: DATASET EXPLORER
# -----------------------------------------------------------------------------
with tab_explore:
    st.subheader("📚 Dataset & Musical Vocabulary")

    meta_file = PROCESSED_DIR / "metadata.json"
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Total Note Events", f"{meta.get('total_notes', 0):,}")
        m_col2.metric("Vocabulary Size", f"{meta.get('vocab_size', 0):,} tokens")
        m_col3.metric("Training Sequences", f"{meta.get('num_sequences', 0):,}")
        m_col4.metric("Sequence Length (Window)", meta.get("sequence_length", 32))

        # Show MIDI files in disk
        c_files = list(CLASSICAL_DIR.glob("*.mid"))
        j_files = list(JAZZ_DIR.glob("*.mid"))

        st.markdown(f"**Collected MIDI Files:** {len(c_files)} Classical pieces | {len(j_files)} Jazz/Blues pieces")

        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            st.markdown("##### 🎻 Classical Collection (Bach, Beethoven, Mozart)")
            st.write([f.name for f in c_files[:10]])
        with sub_col2:
            st.markdown("##### 🎷 Jazz Collection (Ragtime, ii-V-I, 12-Bar Blues)")
            st.write([f.name for f in j_files[:10]])

        # Vocabulary Sample
        st.markdown("##### 🔤 Vocabulary Sample (Notes & Chords)")
        classes = meta.get("classes", [])
        st.write(classes[:40])
    else:
        st.info("No preprocessed dataset metadata found. Run preprocessing from the CLI or training tab.")


# -----------------------------------------------------------------------------
# TAB 3: TRAIN AI MODEL
# -----------------------------------------------------------------------------
with tab_train:
    st.subheader("🧠 Train Deep Learning Model")
    st.markdown("Train the Recurrent Neural Network (LSTM / GRU) on your collected MIDI sequences.")

    train_col1, train_col2 = st.columns(2)
    with train_col1:
        model_arch = st.selectbox("Model Architecture", ["lstm", "gru"], index=0)
        train_epochs = st.number_input("Number of Epochs", min_value=1, max_value=50, value=5)
        train_batch = st.select_slider("Batch Size", options=[64, 128, 256, 512], value=256)
        train_lr = st.select_slider("Learning Rate", options=[0.0005, 0.001, 0.002, 0.005], value=0.002)
        
        start_train_btn = st.button("🚀 Start Training", type="primary")

    with train_col2:
        # Check existing training history
        hist_file = CHECKPOINT_DIR / "history.json"
        if hist_file.exists():
            with open(hist_file, "r", encoding="utf-8") as f:
                history = json.load(f)
            
            st.markdown("##### 📉 Latest Training Loss Curve")
            fig_loss, ax_loss = plt.subplots(figsize=(6, 3.5))
            ax_loss.plot(history["epochs"], history["train_loss"], label="Train Loss", marker="o", color="#3B82F6")
            ax_loss.plot(history["epochs"], history["val_loss"], label="Val Loss", marker="s", color="#EF4444")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("CrossEntropy Loss")
            ax_loss.grid(True, linestyle="--", alpha=0.5)
            ax_loss.legend()
            st.pyplot(fig_loss)
            plt.close(fig_loss)

    if start_train_btn:
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def on_epoch_progress(epoch, total_epochs, train_loss, val_loss):
            progress_bar.progress(epoch / total_epochs)
            status_text.text(f"Epoch {epoch}/{total_epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        with st.spinner("Training neural network..."):
            res = train_model(
                model_type=model_arch,
                epochs=train_epochs,
                batch_size=train_batch,
                learning_rate=train_lr,
                progress_callback=on_epoch_progress
            )
        st.success(f"Training completed! Best Validation Loss: {res['best_val_loss']:.4f}")
        st.rerun()


# -----------------------------------------------------------------------------
# TAB 4: HOW IT WORKS
# -----------------------------------------------------------------------------
with tab_about:
    st.subheader("📖 AI Music Generation Architecture")
    st.markdown("""
    This project generates original musical compositions using deep learning:

    1. **Data Collection**:
       - Extracts classical pieces from Bach, Beethoven, and Mozart via `music21.corpus`.
       - Combines Scott Joplin ragtime with procedural jazz chord progressions (ii-V-I swing, 12-bar blues in F, modal Dorian cool jazz).
    
    2. **music21 Preprocessing**:
       - Converts raw MIDI streams into canonical note, chord, and rest tokens.
       - Maps unique tokens into a vocabulary dictionary.
       - Constructs sliding window sequences ($X_{32} \\to y_{next}$).

    3. **Deep LSTM Neural Network**:
       - **Embedding Layer**: Projects discrete musical tokens into dense 128-d continuous harmonic space.
       - **Multi-layer LSTM**: Learns temporal cadence, voice leading, and melodic patterns.
       - **Dropout & Dense Projection**: Regularizes weights to prevent repetition.

    4. **Sampling & Generation**:
       - **Temperature Scaling**: Regulates creativity versus strict harmonic compliance.
       - **Top-K Filtering**: Truncates unlikely tail tokens for clean phrasing.

    5. **Synthesis & Playback**:
       - Generates standard `.mid` MIDI files.
       - Synthesizes polyphonic `.wav` audio files using acoustic harmonic synthesis and ADSR envelopes.
    """)

    st.markdown("---")
    st.markdown("### 👨‍💻 Developed By")
    st.markdown("""
    - **Developer:** **AASIM KHAN**
    - **Candidate ID:** `CA/DF1/312332`
    """)

