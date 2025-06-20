import torch
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(current_dir, '..', 'train')
sys.path.append(train_dir)

from models import NeuroXVocal

# Define paths
model_path = r"./NeuroXVocal/results/best.pth"
text_file_path = r"./NeuroXVocal/data/.../adrsdt1.txt"
embedding_csv = r"./NeuroXVocal/data/.../audio_embeddings_adrsdt1.csv"
audio_features_csv = r"./NeuroXVocal/data/.../audio_features_adrsdt1.csv"

TEXT_EMBEDDING_MODEL = 'microsoft/deberta-v3-base'  
NUM_MFCC_FEATURES = 47  
NUM_EMBEDDING_FEATURES = 768 

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(TEXT_EMBEDDING_MODEL)
    model = NeuroXVocal(
        num_audio_features=NUM_MFCC_FEATURES,
        num_embedding_features=NUM_EMBEDDING_FEATURES,
        text_embedding_model=TEXT_EMBEDDING_MODEL,
    )

    model.to(device)
    state_dict = torch.load(model_path, map_location=device)
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
    with open(text_file_path, 'r') as file:
        text = file.read()
    text_tokens = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt',
    )
    text_tokens = {key: value.to(device) for key, value in text_tokens.items()}
    audio_features_df = pd.read_csv(audio_features_csv)
    audio_features = audio_features_df.drop(columns=['patient_id']).iloc[0].values.astype(float)
    audio_tensor = torch.tensor(audio_features, dtype=torch.float32).unsqueeze(0).to(device)  
    embedding_features_df = pd.read_csv(embedding_csv)
    embedding_features = embedding_features_df.drop(columns=['patient_id']).iloc[0].values.astype(float)
    embedding_tensor = torch.tensor(embedding_features, dtype=torch.float32).unsqueeze(0).to(device)  
    with torch.no_grad():
        outputs = model(text_tokens, audio_tensor, embedding_tensor)
        probabilities = torch.sigmoid(outputs)
        confidence_score = probabilities.item()
        predicted_class = 1 if confidence_score > 0.5 else 0

    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence Score: {confidence_score:.4f}")

if __name__ == '__main__':
    main()
