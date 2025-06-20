import pandas as pd
import os
from pathlib import Path
import numpy as np
from typing import Dict, Tuple, List

"""
This module handles loading and preprocessing of patient data, audio features,
and literature for the Alzheimer's explainer system.
"""

class DataLoader:
    def __init__(self):
        self.ad_path = Path(r"./NeuroXVocal/data/.../ad")
        self.cn_path = Path(r"./NeuroXVocal/data/.../cn")
        self.literature_path = Path(r"./NeuroXVocal/src/explainer/literature")
        for path in [self.ad_path, self.cn_path, self.literature_path]:
            if not path.exists():
                raise FileNotFoundError(f"Directory not found: {path}")
    
    def load_audio_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load audio features for both AD and CN patients."""
        ad_features_file = self.ad_path / "audio_features_ad.csv"
        cn_features_file = self.cn_path / "audio_features_cn.csv"
        
        if not ad_features_file.exists():
            raise FileNotFoundError(f"Audio features file not found: {ad_features_file}")
        if not cn_features_file.exists():
            raise FileNotFoundError(f"Audio features file not found: {cn_features_file}")
        
        ad_features = pd.read_csv(ad_features_file)
        cn_features = pd.read_csv(cn_features_file)
        return ad_features, cn_features
    
    def load_transcriptions(self) -> Dict[str, str]:
        """Load all patient transcriptions into a dictionary."""
        transcriptions = {}
        for txt_file in self.ad_path.glob("*.txt"):
            if txt_file.stem != "audio_features_ad":
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        transcriptions[txt_file.stem] = content
                    else:
                        print(f"Warning: {txt_file} is empty.")
        for txt_file in self.cn_path.glob("*.txt"):
            if txt_file.stem != "audio_features_cn":
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        transcriptions[txt_file.stem] = content
                    else:
                        print(f"Warning: {txt_file} is empty.")
                        
        return transcriptions
    
    def load_literature(self) -> List[str]:
        """Load literature review documents."""
        literature = []
        txt_files = list(self.literature_path.glob("*.txt"))
        print(f"Found {len(txt_files)} literature files in {self.literature_path}")
        
        for txt_file in txt_files:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    literature.append(content)
                else:
                    print(f"Warning: {txt_file} is empty.")
        
        print(f"Loaded {len(literature)} literature documents.")
        return literature
    
    def get_patient_data(self, patient_id: str) -> Dict:
        """Get all data for a specific patient."""
        ad_transcription_file = self.ad_path / f"{patient_id}.txt"
        if ad_transcription_file.exists():
            is_ad = True
            features_csv = self.ad_path / "audio_features_ad.csv"
        else:
            cn_transcription_file = self.cn_path / f"{patient_id}.txt"
            if cn_transcription_file.exists():
                is_ad = False
                features_csv = self.cn_path / "audio_features_cn.csv"
            else:
                raise FileNotFoundError(f"Transcription file for patient ID {patient_id} not found in both AD and CN directories.")
        
        if not features_csv.exists():
            raise FileNotFoundError(f"Audio features file not found: {features_csv}")
        
        features_df = pd.read_csv(features_csv)
        
        if patient_id not in features_df['patient_id'].values:
            raise ValueError(f"Patient ID {patient_id} not found in {features_csv}")
        
        patient_features = features_df[features_df['patient_id'] == patient_id].iloc[0]
        
        # Load transcription
        transcription_file = ad_transcription_file if is_ad else cn_transcription_file
        with open(transcription_file, 'r', encoding='utf-8') as f:
            transcription = f.read().strip()
            if not transcription:
                print(f"Warning: {transcription_file} is empty.")
        
        return {
            'patient_id': patient_id,
            'class': 'AD' if is_ad else 'CN',
            'features': patient_features,
            'transcription': transcription
        }
