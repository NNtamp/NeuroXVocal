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
    query = f"""
    Speech patterns and indicators related to:
    - Pauses: {patient_data['features']['speech_pause_ratio']}, {patient_data['features']['num_pauses']}
    - Speech rate: {patient_data['features']['speaking_rate']}, {patient_data['features']['articulation_rate']}
    - Voice characteristics: pitch {patient_data['features']['pitch_mean']}, intensity {patient_data['features']['intensity_mean']}
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
