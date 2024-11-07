import torch
from transformers import AutoTokenizer
import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np
import pandas as pd
from pathlib import Path
import os
from PIL import Image
import time
from datetime import datetime
import subprocess
import sys
import shutil
import whisper
from constants import *


if TRAIN_DIR not in sys.path:
    sys.path.append(TRAIN_DIR)

from models import NeuroXVocal


def load_image(image_path):
    """Load and display an image in Streamlit"""
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

def create_patient_folder(recordings_path):
    """Create a unique folder for each recording session with a timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    patient_folder = recordings_path / f"patient_{timestamp}"
    patient_folder.mkdir(exist_ok=True)
    st.session_state.current_folder = patient_folder
    return patient_folder

def start_recording(sample_rate):
    """Starts recording audio."""
    st.session_state.is_recording = True
    st.session_state.recording_completed = False
    st.session_state.transcription_completed = False
    st.session_state.record_start_time = time.time()
    
    duration = 3600  # maximum duration 1 hour
    st.session_state.audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    st.info("Recording in progress... Press START/STOP to end.")

def stop_recording(recordings_path, sample_rate):
    """Stops audio recording, trims, and saves the data to a file."""
    sd.stop()
    st.session_state.is_recording = False
    end_time = time.time()
    recorded_duration = end_time - st.session_state.record_start_time
    num_frames = int(recorded_duration * sample_rate)
    audio_data = st.session_state.audio_data[:num_frames]
    patient_folder = create_patient_folder(recordings_path)
    file_path = patient_folder / "description.wav"
    sf.write(file_path, audio_data, sample_rate)
    st.session_state.current_audio_path = str(file_path)
    st.session_state.recording_completed = True
    st.success("Recording saved successfully!")
    st.session_state.record_start_time = None

def extract_audio_embeddings(audio_folder):
    """Extracts audio embeddings using the external script"""
    try:
        script_path = r"C:/Users/30697/Desktop/Personal Projects/NeuroXVocal/src/data_extraction/extract_audio_embeddings.py"
        result = subprocess.run([
            sys.executable, script_path, audio_folder,
            '--output_csv', os.path.join(audio_folder, 'audio_embeddings.csv')
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            st.error(f"Error extracting audio embeddings: {result.stderr}")
    except Exception as e:
        st.error(f"Error during embeddings extraction: {str(e)}")

def process_audio_embeddings(embeddings_path):
    """Processes audio embeddings using the preprocessing script"""
    try:
        script_path = r"C:/Users/30697/Desktop/Personal Projects/NeuroXVocal/src/data_processing/preprocess_audio_emb.py"
        scaler_path = r"C:/Users/30697/Desktop/Personal Projects/NeuroXVocal/src/inference/scaler_params_audio_emb.pkl"
        output_path = os.path.join(os.path.dirname(embeddings_path), 'audio_embeddings_processed.csv')
        
        result = subprocess.run([
            sys.executable, script_path, embeddings_path, scaler_path, output_path
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            st.error(f"Error processing audio embeddings: {result.stderr}")
    except Exception as e:
        st.error(f"Error during embeddings processing: {str(e)}")

def create_llm_features(csv_path):
    """Creates a simplified version of audio features for LLM"""
    try:
        df = pd.read_csv(csv_path)
        df = df.drop(['jitter_local', 'shimmer_local', 'formant_1_mean', 
                     'formant_1_std', 'formant_2_mean', 'formant_2_std', 
                     'formant_3_mean', 'formant_3_std', 'class'], axis=1)
        
        output_path = os.path.join(os.path.dirname(csv_path), 'audio_features_llm.csv')
        df.to_csv(output_path, index=False)
    except Exception as e:
        st.error(f"Error creating LLM features: {str(e)}")

def process_audio_features(csv_path):
    """Processes audio features using the external preprocessing script"""
    try:
        script_path = r"C:/Users/30697/Desktop/Personal Projects/NeuroXVocal/src/data_processing/preprocess_audio_features.py"
        output_dir = os.path.dirname(csv_path)
        scaler_path = r"C:/Users/30697/Desktop/Personal Projects/NeuroXVocal/src/inference/scaler_params_audio_features.pkl"
        
        temp_output_dir = os.path.join(output_dir, 'temp_processed')
        os.makedirs(temp_output_dir, exist_ok=True)
        
        result = subprocess.run([
            sys.executable, script_path,
            "--input_path", csv_path,
            "--output_path", temp_output_dir,
            "--scaler_path", scaler_path
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            processed_file = os.path.join(temp_output_dir, os.path.basename(csv_path))
            if os.path.exists(processed_file):
                shutil.move(processed_file, os.path.join(output_dir, 'audio_features_processed.csv'))
        else:
            st.error(f"Error processing audio features: {result.stderr}")
            
        shutil.rmtree(temp_output_dir, ignore_errors=True)
    except Exception as e:
        st.error(f"Error during feature processing: {str(e)}")

def process_text(text_path):
    """Processes text using the external preprocessing script"""
    try:
        script_path = r"C:/Users/30697/Desktop/Personal Projects/NeuroXVocal/src/data_processing/preprocess_texts.py"
        input_dir = os.path.dirname(text_path)
        
        temp_output_dir = os.path.join(input_dir, 'temp_processed')
        os.makedirs(temp_output_dir, exist_ok=True)
        
        result = subprocess.run([
            sys.executable, script_path, input_dir, temp_output_dir
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            processed_file = os.path.join(temp_output_dir, os.path.basename(text_path))
            if os.path.exists(processed_file):
                new_name = os.path.join(input_dir, 'description_processed.txt')
                shutil.move(processed_file, new_name)
        else:
            st.error(f"Error processing text: {result.stderr}")
            
        shutil.rmtree(temp_output_dir, ignore_errors=True)
    except Exception as e:
        st.error(f"Error during text processing: {str(e)}")

def analyze_audio():
    """Performs transcription, feature extraction, and embeddings extraction/processing"""
    try:
        audio_folder = os.path.dirname(st.session_state.current_audio_path)
        
        with st.spinner("Transcribing audio..."):
            model = whisper.load_model("base")
            result = model.transcribe(
                st.session_state.current_audio_path,
                language="en",
                task="transcribe",
                fp16=False,
                initial_prompt="This is a recording describing an image in English."
            )
            
            transcript_path = os.path.join(audio_folder, 'description.txt')
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(result["text"])
            process_text(transcript_path)
            st.text("Transcription preview:")
            st.write(result["text"])
        
        with st.spinner("Processing audio..."):
            feature_script = r"C:/Users/30697/Desktop/Personal Projects/NeuroXVocal/src/data_extraction/extract_audio_features.py"
            features_path = os.path.join(audio_folder, 'audio_features.csv')
            
            subprocess.run([
                sys.executable,
                feature_script,
                audio_folder,
                '--output_csv', features_path
            ], capture_output=True, text=True)
            create_llm_features(features_path)
            process_audio_features(features_path)
            extract_audio_embeddings(audio_folder)
            embeddings_path = os.path.join(audio_folder, 'audio_embeddings.csv')
            if os.path.exists(embeddings_path):
                process_audio_embeddings(embeddings_path)
            
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")

def get_prediction(text_path, audio_features_path, embeddings_path):
    """
    Predicts dementia likelihood using processed text, audio features, and embeddings
    """
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained(TEXT_EMBEDDING_MODEL)
        model = NeuroXVocal(
            num_audio_features=NUM_MFCC_FEATURES,
            num_embedding_features=NUM_EMBEDDING_FEATURES,
            text_embedding_model=TEXT_EMBEDDING_MODEL,
        )
        model.to(device)
        state_dict = torch.load(MODEL_PATH, map_location=device)
        if 'module.' in list(state_dict.keys())[0]:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.', '')
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
        model.eval()

        # Process text
        with open(text_path, 'r') as file:
            text = file.read()
        text_tokens = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt',
        )
        text_tokens = {key: value.to(device) for key, value in text_tokens.items()}

        audio_features_df = pd.read_csv(audio_features_path)
        audio_features = audio_features_df.drop(columns=['patient_id']).iloc[0].values.astype(float)
        audio_tensor = torch.tensor(audio_features, dtype=torch.float32).unsqueeze(0).to(device)
        embedding_features_df = pd.read_csv(embeddings_path)
        embedding_features = embedding_features_df.drop(columns=['patient_id']).iloc[0].values.astype(float)
        embedding_tensor = torch.tensor(embedding_features, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(text_tokens, audio_tensor, embedding_tensor)
            probabilities = torch.sigmoid(outputs)
            confidence_score = probabilities.item()
            predicted_class = 1 if confidence_score > 0.5 else 0

        return predicted_class, confidence_score

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        raise 
        return None, None

def generate_prediction_message(predicted_class, confidence_score):
    """
    Generates a detailed message based on the prediction results
    """
    if predicted_class is None or confidence_score is None:
        return "I apologize, but I was unable to make a prediction due to an error in processing the data."

    confidence_percentage = confidence_score * 100
    
    if predicted_class == 1:
        risk_level = "high" if confidence_percentage > 75 else "moderate"
        message = f"""
        Based on my analysis of the speech patterns, linguistic features, and audio characteristics, 
        I detect indicators that suggest a {risk_level} likelihood of cognitive impairment 
        consistent with dementia and Alzheimer's disease. 

        Please note that this is not a clinical diagnosis, but rather an analytical assessment 
        based on speech and language patterns. I strongly recommend consulting with a healthcare 
        professional for a comprehensive evaluation.

        Key points to remember:
        • This is an AI-based screening tool
        • Only a qualified healthcare provider can make a formal diagnosis
        • Early consultation with medical professionals is beneficial
        """
    else:
        message = f"""
        Based on my analysis of the speech patterns, linguistic features, and audio characteristics,
        I do not detect significant indicators of cognitive impairment.

        However, please remember:
        • This is an AI-based screening tool
        • Regular check-ups with healthcare providers are important
        • If you have concerns, consult with a medical professional
        """
    
    return message.strip()