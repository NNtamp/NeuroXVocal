import torch
import torch.nn as nn
from transformers import ElectraModel


class MultiModalDementiaClassifier(nn.Module):
    def __init__(self, num_audio_features, text_embedding_model, audio_length=47):
        super(MultiModalDementiaClassifier, self).__init__()
        self.text_model = ElectraModel.from_pretrained(text_embedding_model)
        self.text_fc = nn.Linear(self.text_model.config.hidden_size, 256)
        self.text_dropout = nn.Dropout(0.3)
        self.text_layer_norm = nn.LayerNorm(256)
        self.audio_cnn = nn.Sequential(
            nn.Conv1d(in_channels=num_audio_features, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(256 * (audio_length // 4), 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(256)
        )
        self.cross_attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, text_input, audio_input):
        input_ids = text_input['input_ids']
        attention_mask = text_input['attention_mask']
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = text_outputs.last_hidden_state
        text_features = self.text_fc(text_embeddings)
        text_features = self.text_dropout(text_features)
        text_features = self.text_layer_norm(text_features)
        text_features = text_features.permute(1, 0, 2)
        attn_output, _ = self.cross_attention(text_features, text_features, text_features)
        attn_output = attn_output.permute(1, 0, 2)
        text_attn_pooled = torch.mean(attn_output, dim=1)
        audio_features = self.audio_cnn(audio_input)
        combined_features = torch.cat((text_attn_pooled, audio_features), dim=1)
        output = self.classifier(combined_features)
        return output
