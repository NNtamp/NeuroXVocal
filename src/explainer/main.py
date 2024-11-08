from data_loader import DataLoader
from vector_store import VectorStore
from prompt_builder import PromptBuilder
from llm_explainer import LLMExplainer
import argparse

"""
Main script that ties together all components of the explainer system.
"""

def main():
    parser = argparse.ArgumentParser(description='Alzheimer\'s Patient Explainer')
    parser.add_argument('--patient_id', required=True, help='Patient ID to analyze')
    args = parser.parse_args()
    
    try:
        data_loader = DataLoader()
    except FileNotFoundError as e:
        print(f"Error initializing DataLoader: {e}")
        return
    except Exception as e:
        print(f"Unexpected error initializing DataLoader: {e}")
        return
    
    vector_store = VectorStore()
    prompt_builder = PromptBuilder()
    llm_explainer = LLMExplainer()
    
    try:
        literature = data_loader.load_literature()
        if not literature:
            print("No literature documents loaded. Exiting.")
            return
    except Exception as e:
        print(f"Error loading literature: {e}")
        return
    
    try:
        vector_store.create_literature_index(literature)
    except Exception as e:
        print(f"Error creating literature index: {e}")
        return
    
    try:
        patient_data = data_loader.get_patient_data(args.patient_id)
    except FileNotFoundError as e:
        print(f"Error fetching patient data: {e}")
        return
    except ValueError as e:
        print(f"Data error: {e}")
        return
    except Exception as e:
        print(f"Unexpected error fetching patient data: {e}")
        return

    # Enhanced query with comprehensive acoustic features aligned with CSV columns
    query = f"""
    Speech patterns and acoustic features related to cognitive decline indicators in dementia and Alzheimer's Disease:

    1. Temporal Characteristics:
       - Recording duration: {patient_data['features']['duration']:.2f} seconds
       - Total speech time: {patient_data['features']['total_speech_time']:.2f} seconds
       
    2. Pause Patterns:
       - Speech-pause ratio: {patient_data['features']['speech_pause_ratio']:.3f}
       - Number of pauses: {patient_data['features']['num_pauses']:.0f}
       - Total pause duration: {patient_data['features']['total_pause_duration']:.3f} seconds
       - Average pause duration: {patient_data['features']['avg_pause_duration']:.3f} seconds
       - Maximum pause duration: {patient_data['features']['max_pause_duration']:.3f} seconds
       - Pause duration std: {patient_data['features']['pause_duration_std']:.3f}
    
    3. Speech Rate Metrics:
       - Speaking rate: {patient_data['features']['speaking_rate']:.2f} syllables/second
       - Articulation rate: {patient_data['features']['articulation_rate']:.2f} syllables/second
    
    4. Voice Characteristics:
       - Pitch measurements:
         * Mean: {patient_data['features']['pitch_mean']:.1f} Hz
         * Standard deviation: {patient_data['features']['pitch_std']:.1f} Hz
         * Range: {patient_data['features']['pitch_range']:.1f} Hz
       - Intensity measurements:
         * Mean: {patient_data['features']['intensity_mean']:.1f} dB
         * Standard deviation: {patient_data['features']['intensity_std']:.1f} dB
         * Range: {patient_data['features']['intensity_range']:.1f} dB
       - Spectral characteristics:
         * Centroid mean: {patient_data['features']['spectral_centroid_mean']:.1f}
         * Centroid std: {patient_data['features']['spectral_centroid_std']:.1f}
       - Voice quality indicators:
         * Zero-crossing rate mean: {patient_data['features']['zcr_mean']:.3f}
         * Zero-crossing rate std: {patient_data['features']['zcr_std']:.3f}
         * Harmonics-to-noise ratio: {patient_data['features']['hnr_mean']:.2f}
    
    5. Acoustic Texture (MFCCs):
       - MFCC 1: mean {patient_data['features']['mfcc_1_mean']:.3f}, std {patient_data['features']['mfcc_1_std']:.3f}
       - MFCC 2: mean {patient_data['features']['mfcc_2_mean']:.3f}, std {patient_data['features']['mfcc_2_std']:.3f}
       - MFCC 3: mean {patient_data['features']['mfcc_3_mean']:.3f}, std {patient_data['features']['mfcc_3_std']:.3f}
       - MFCC 4: mean {patient_data['features']['mfcc_4_mean']:.3f}, std {patient_data['features']['mfcc_4_std']:.3f}
       - MFCC 5: mean {patient_data['features']['mfcc_5_mean']:.3f}, std {patient_data['features']['mfcc_5_std']:.3f}
       - MFCC 6: mean {patient_data['features']['mfcc_6_mean']:.3f}, std {patient_data['features']['mfcc_6_std']:.3f}
       - MFCC 7: mean {patient_data['features']['mfcc_7_mean']:.3f}, std {patient_data['features']['mfcc_7_std']:.3f}
       - MFCC 8: mean {patient_data['features']['mfcc_8_mean']:.3f}, std {patient_data['features']['mfcc_8_std']:.3f}
       - MFCC 9: mean {patient_data['features']['mfcc_9_mean']:.3f}, std {patient_data['features']['mfcc_9_std']:.3f}
       - MFCC 10: mean {patient_data['features']['mfcc_10_mean']:.3f}, std {patient_data['features']['mfcc_10_std']:.3f}
       - MFCC 11: mean {patient_data['features']['mfcc_11_mean']:.3f}, std {patient_data['features']['mfcc_11_std']:.3f}
       - MFCC 12: mean {patient_data['features']['mfcc_12_mean']:.3f}, std {patient_data['features']['mfcc_12_std']:.3f}
       - MFCC 13: mean {patient_data['features']['mfcc_13_mean']:.3f}, std {patient_data['features']['mfcc_13_std']:.3f}
    """

    relevant_literature = vector_store.get_relevant_literature(query)
    if not relevant_literature:
        print("No relevant literature found for the given query.")
        return
    prompt = prompt_builder.create_prompt(patient_data, relevant_literature)
    try:
        explanation = llm_explainer.generate_explanation(prompt)
    except Exception as e:
        print(f"Error generating explanation: {e}")
        return
    print(f"\nExplanation for Patient {args.patient_id}:")
    print("-" * 80)
    print(explanation)

if __name__ == "__main__":
    main()