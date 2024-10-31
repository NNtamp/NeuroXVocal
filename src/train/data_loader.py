import torch
from torch.utils.data import Dataset, ConcatDataset
from transformers import AutoTokenizer
import pandas as pd
import os
from config import TEXT_EMBEDDING_MODEL

class DementiaDataset(Dataset):
    def __init__(
        self,
        audio_csv_path,
        embedding_csv_path,
        text_dir,
        label_value=None,
        tokenizer_model=TEXT_EMBEDDING_MODEL,
        max_length=512
    ):
        self.audio_data = pd.read_csv(audio_csv_path)
        self.embedding_data = pd.read_csv(embedding_csv_path)
        self.text_dir = text_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.max_length = max_length
        self.data = pd.merge(
            self.audio_data,
            self.embedding_data,
            on='patient_id',
            suffixes=('_audio', '_embedding')
        )

        self.label_value = label_value
        self.data['label'] = self.label_value

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_row = self.data.iloc[idx]
        patient_id = data_row['patient_id']
        text_file_path = os.path.join(self.text_dir, f'{patient_id}.txt')

        with open(text_file_path, 'r') as file:
            text = file.read()

        text_tokens = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        
        audio_feature_columns = [
            col for col in self.audio_data.columns if col != 'patient_id'
        ]
        embedding_feature_columns = [
            col for col in self.embedding_data.columns if col != 'patient_id'
        ]
        audio_features = data_row[audio_feature_columns].values.astype(float)
        embedding_features = data_row[embedding_feature_columns].values.astype(float)
        label = torch.tensor(data_row['label'], dtype=torch.float32)

        audio_tensor = torch.tensor(audio_features, dtype=torch.float32)
        embedding_tensor = torch.tensor(embedding_features, dtype=torch.float32)

        return text_tokens, audio_tensor, embedding_tensor, label

def create_full_dataset(
    ad_text_dir,
    cn_text_dir,
    ad_csv,
    cn_csv,
    ad_embedding_csv,
    cn_embedding_csv
):
    ad_dataset = DementiaDataset(
        ad_csv,
        ad_embedding_csv,
        ad_text_dir,
        label_value=1
    )
    cn_dataset = DementiaDataset(
        cn_csv,
        cn_embedding_csv,
        cn_text_dir,
        label_value=0
    )
    full_dataset = ConcatDataset([ad_dataset, cn_dataset])
    return full_dataset
