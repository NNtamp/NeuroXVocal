from typing import Dict, List

"""
Updated prompt builder with consistent response format starting with patient classification.
"""


class PromptBuilder:
    def __init__(self):
        self.base_prompt = """
        Task: Analyze the following patient data and explain why they likely have or don't have Alzheimer's disease.
        Start your response with exactly: "{opening_sentence}"
        Then provide a detailed explanation.
        
        Patient Information:
        - ID: {patient_id}
        - Current Diagnosis: {diagnosis}
        
        Key Speech Features:
        {audio_features}
        
        Patient's Speech Transcription:
        "{transcription}"
        
        Relevant Research Findings:
        {literature}
        
        Remember to start with the exact opening sentence provided, then explain why this patient {has_or_not} Alzheimer's disease. 
        Connect the observed features with the research findings and speech patterns.
        Focus on the most significant indicators in both the audio features and transcription.
        """
        
    def create_prompt(self, 
                     patient_data: Dict, 
                     relevant_literature: List[str]) -> str:
        """Create a formatted prompt for FLAN-T5."""
        opening_sentence = ("The patient is probably an Alzheimer sample." 
                          if patient_data['class'] == 'AD' 
                          else "The patient is probably a Healthy sample.")
        key_features = [
            'speech_pause_ratio', 'num_pauses', 'avg_pause_duration',
            'pitch_mean', 'intensity_mean', 'articulation_rate',
            'speaking_rate'
        ]
        formatted_features = "\n".join([
            f"- {feature.replace('_', ' ').title()}: {patient_data['features'][feature]:.3f}"
            for feature in key_features
        ])
        formatted_literature = "\n".join([
            f"- {lit.strip()}" for lit in relevant_literature
        ])
        
        return self.base_prompt.format(
            opening_sentence=opening_sentence,
            patient_id=patient_data['patient_id'],
            diagnosis=patient_data['class'],
            audio_features=formatted_features,
            transcription=patient_data['transcription'],
            literature=formatted_literature,
            has_or_not="has" if patient_data['class'] == 'AD' else "doesn't have"
        )
