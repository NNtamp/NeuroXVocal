import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
import os
from PIL import Image
import time
from datetime import datetime

# Set up the base recordings directory
RECORDINGS_PATH = Path("C:/Users/30697/Desktop/Personal Projects/NeuroXVocal/app/recordings")
RECORDINGS_PATH.mkdir(exist_ok=True)


if 'patient_number' not in st.session_state:
    st.session_state.patient_number = 1
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'record_start_time' not in st.session_state:
    st.session_state.record_start_time = None


st.set_page_config(page_title="NeuroXVocal Machine", layout="centered")


st.markdown(
    """
    <style>
    body {
        background-color: #000000;
        color: #FFFFFF;
    }
    .title {
        font-size: 48px;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: bold;
        margin-top: 20px;
    }
    .instruction {
        font-size: 18px;
        text-align: center;
        margin-bottom: 30px;
        color: #FFFFFF;
    }
    .instruction .start-word {
        font-size: 24px;
        font-weight: bold;
        color: white; /* Change START/STOP color to white */
    }
    .reference {
        font-size: 12px;
        color: #CCCCCC;
        margin-top: 10px;
        text-align: center;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown('<div class="title">NeuroXVocal Machine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="instruction">Please press the <span class="start-word">START/STOP</span> button and describe loud and clear the below image.</div>',
    unsafe_allow_html=True
)


IMAGE_PATH = "image/cookie_theft.jpg"

def load_image(image_path):
    if os.path.exists(image_path):
        try:
            image = Image.open(image_path)
            st.image(image, use_column_width=True)
            return True
        except:
            st.error("Error loading image.")
            return False
    else:
        st.error("Image not found at the specified path.")
        return False

image_loaded = load_image(IMAGE_PATH)

if image_loaded:
    reference_text = "Goodglass, H., Kaplan, E., & Barresi, B. (2001). Boston Diagnostic Aphasia Examination–Third Edition (BDAE-3). Baltimore, MD: Lippincott Williams & Wilkins."
    st.markdown(f'<div class="reference">Reference: {reference_text}</div>', unsafe_allow_html=True)
else:
    st.info("Please ensure the image path is correct.")


SAMPLE_RATE = 16000  # Hz

def create_patient_folder():
    """Create a unique folder for each recording session with a timestamp."""
    # timestamp in the format YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    patient_folder = RECORDINGS_PATH / f"patient_{timestamp}"
    patient_folder.mkdir(exist_ok=True)
    st.session_state.current_folder = patient_folder
    return patient_folder

def start_recording():
    """Starts recording audio."""
    st.session_state.is_recording = True
    st.session_state.record_start_time = time.time()  

    duration = 3600  # maximum duration 1 hour
    st.session_state.audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    st.info("Recording in progress... Press START/STOP to end.")

def stop_recording():
    """Stops audio recording, trims, and saves the data to a file."""
    sd.stop()
    st.session_state.is_recording = False
    end_time = time.time()
    recorded_duration = end_time - st.session_state.record_start_time
    num_frames = int(recorded_duration * SAMPLE_RATE)
    audio_data = st.session_state.audio_data[:num_frames]
    patient_folder = create_patient_folder()
    file_path = patient_folder / "description.wav"
    sf.write(file_path, audio_data, SAMPLE_RATE)
    st.success("Recording saved successfully!")
    st.session_state.record_start_time = None
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    if st.button("START/STOP", use_container_width=True):
        if st.session_state.is_recording:
            stop_recording()
        else:
            start_recording()

st.markdown(
    """
    ---
    <div style="text-align: center; font-size: 10px; color: #888888;">
        NeuroXVocal Machine &copy; 2025
    </div>
    """,
    unsafe_allow_html=True
)
