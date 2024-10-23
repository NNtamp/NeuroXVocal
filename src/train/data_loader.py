import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer

class DementiaDataset(Dataset):
    def __init__(self, audio_csv_path, text_dir, label_value, tokenizer_model='bert-base-uncased', max_length=512):
        print(f"Attempting to read CSV at: {audio_csv_path}")
        self.audio_data = pd.read_csv(audio_csv_path)
        self.text_dir = text_dir
        self.label_value = label_value  # (0 for cn, 1 for ad)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.max_length = max_length

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, idx):
        audio_row = self.audio_data.iloc[idx]
        patient_id = audio_row['patient_id']
        text_file_path = os.path.join(self.text_dir, f'{patient_id}.txt')
        with open(text_file_path, 'r') as file:
            text = file.read()
        text_tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        audio_features = audio_row.drop(labels=['patient_id']).values.astype(float)
        audio_tensor = torch.tensor(audio_features, dtype=torch.float32)

        label = torch.tensor(self.label_value, dtype=torch.float32)

        return text_tokens, audio_tensor, label

def create_dataloaders(batch_size, ad_text_dir, cn_text_dir, ad_csv, cn_csv, k_folds=5):
    ad_dataset = DementiaDataset(ad_csv, ad_text_dir, label_value=1)
    cn_dataset = DementiaDataset(cn_csv, cn_text_dir, label_value=0)
    
    full_dataset = torch.utils.data.ConcatDataset([ad_dataset, cn_dataset])

    dataloaders = []
    for fold in range(k_folds):
        train_loader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
        dataloaders.append((train_loader, val_loader))
        
    return dataloaders



