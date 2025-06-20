import streamlit as st
from pathlib import Path
import os
from utils import *
from constants import *


RECORDINGS_PATH = Path("/NeuroXVocal/app/recordings")
RECORDINGS_PATH.mkdir(exist_ok=True)


if 'patient_number' not in st.session_state:
    st.session_state.patient_number = 1
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'record_start_time' not in st.session_state:
    st.session_state.record_start_time = None
if 'recording_completed' not in st.session_state:
    st.session_state.recording_completed = False
if 'current_audio_path' not in st.session_state:
    st.session_state.current_audio_path = None
if 'explanation_generated' not in st.session_state:
    st.session_state.explanation_generated = False

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
        color: white;
    }
    .reference {
        font-size: 12px;
        color: #CCCCCC;
        margin-top: 10px;
        text-align: center;
        font-style: italic;
    }
    .stButton>button {
        width: 100%;
        margin: 5px 0;
        height: 3em;
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
SAMPLE_RATE = 16000  # Hz

image_loaded = load_image(IMAGE_PATH)

if image_loaded:
    reference_text = "Goodglass, H., Kaplan, E., & Barresi, B. (2001). Boston Diagnostic Aphasia Examination–Third Edition (BDAE-3). Baltimore, MD: Lippincott Williams & Wilkins."
    st.markdown(f'<div class="reference">Reference: {reference_text}</div>', unsafe_allow_html=True)
else:
    st.info("Please ensure the image path is correct.")

col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    if st.button("START/STOP", use_container_width=True, key="record_button"):
        if st.session_state.is_recording:
            stop_recording(RECORDINGS_PATH, SAMPLE_RATE)
        else:
            start_recording(SAMPLE_RATE)
    
    if st.session_state.recording_completed:
        if st.button("START ANALYSIS", use_container_width=True, key="analysis_button", type="secondary"):
            analyze_audio()

if st.session_state.get('current_audio_path'):
    audio_folder = os.path.dirname(st.session_state.current_audio_path)
    text_path = os.path.join(audio_folder, 'description_processed.txt')
    audio_features_path = os.path.join(audio_folder, 'audio_features_processed.csv')
    embeddings_path = os.path.join(audio_folder, 'audio_embeddings_processed.csv')
    
    if all(os.path.exists(f) for f in [text_path, audio_features_path, embeddings_path]):
        st.markdown("---")
        st.markdown("### AI Analysis Results")
        
        chat_container = st.container()
        
        with chat_container:
            with st.spinner("Analyzing speech patterns and generating assessment..."):
                predicted_class, confidence_score = get_prediction(
                    text_path, audio_features_path, embeddings_path
                )
                
                if predicted_class is not None:
                    message = generate_prediction_message(predicted_class, confidence_score)
                    st.markdown(
                        """
                        <div style='background-color: #1E1E1E; padding: 20px; border-radius: 10px; margin: 10px 0;'>
                            <p style='color: #FFFFFF; margin: 0;'>""" + message.replace('\n', '<br>') + """</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    if not st.session_state.explanation_generated:
                        st.markdown("### Detailed AI Explanation")
                        with st.spinner("Generating detailed analysis explanation..."):
                            try:
                                data_loader = DataLoader()
                                vector_store = VectorStore()
                                prompt_builder = PromptBuilder()
                                llm_explainer = LLMExplainer()
                                literature = data_loader.load_literature()
                                vector_store.create_literature_index(literature)
                                patient_features = pd.read_csv(os.path.join(audio_folder, 'audio_features_llm.csv')).iloc[0]
                                with open(text_path, 'r') as f:
                                    transcription = f.read()

                                patient_data = {
                                    'patient_id': 'current',
                                    'class': 'AD' if predicted_class == 1 else 'CN',
                                    'features': patient_features,
                                    'transcription': transcription
                                }
                                query = create_feature_query(patient_features)
                                relevant_literature = vector_store.get_relevant_literature(query)
                                prompt = prompt_builder.create_prompt(patient_data, relevant_literature)
                                explanation = llm_explainer.generate_explanation(prompt)

                                st.markdown(
                                    """
                                    <div style='background-color: #1E1E1E; padding: 20px; border-radius: 10px; margin: 10px 0;'>
                                        <p style='color: #FFFFFF; margin: 0;'>""" + explanation.replace('\n', '<br>') + """</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                st.session_state.explanation_generated = True
                            except Exception as e:
                                st.error(f"Error generating explanation: {str(e)}")

st.markdown(
    """
    ---
    <div style="text-align: center; font-size: 10px; color: #888888;">
        NeuroXVocal Machine &copy; 2025
    </div>
    """,
    unsafe_allow_html=True
)