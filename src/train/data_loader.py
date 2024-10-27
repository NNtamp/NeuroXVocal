import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold
from config import TEXT_EMBEDDING_MODEL

class DementiaDataset(Dataset):

    def __init__(self, audio_csv_path, text_dir, label_value, tokenizer_model=TEXT_EMBEDDING_MODEL, max_length=512):
        print(f"Attempting to read CSV at: {audio_csv_path}")
        self.audio_data = pd.read_csv(audio_csv_path)
        self.text_dir = text_dir
        self.label_value = label_value
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_EMBEDDING_MODEL)  # Adjusts to new model
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
        audio_tensor = torch.tensor(audio_features, dtype=torch.float32).view(1,47)

        label = torch.tensor(self.label_value, dtype=torch.float32)

        return text_tokens, audio_tensor, label

def create_dataloaders(batch_size, ad_text_dir, cn_text_dir, ad_csv, cn_csv, k_folds=5):
    ad_dataset = DementiaDataset(ad_csv, ad_text_dir, label_value=1)
    cn_dataset = DementiaDataset(cn_csv, cn_text_dir, label_value=0)
    
    full_dataset = torch.utils.data.ConcatDataset([ad_dataset, cn_dataset])
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    
    labels = []
    for dataset in [ad_dataset, cn_dataset]:
        for idx in range(len(dataset)):
            _, _, label = dataset[idx]
            labels.append(int(label.item()))
    
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    dataloaders = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices, labels)):
        print(f"Creating data loaders for fold {fold + 1}")
        
        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_dataset, val_idx)

        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        dataloaders.append((train_loader, val_loader))
        
    return dataloaders
        
