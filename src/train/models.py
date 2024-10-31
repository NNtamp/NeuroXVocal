import torch
import torch.nn as nn
from transformers import DebertaV2Model 

class NeuroXVocal(nn.Module):
    def __init__(
        self,
        num_audio_features,
        num_embedding_features,
        text_embedding_model,
        hidden_size=768
    ):
        super(NeuroXVocal, self).__init__()

        self.text_model = DebertaV2Model.from_pretrained(text_embedding_model)
        self.hidden_size = self.text_model.config.hidden_size
        self.audio_fc = nn.Sequential(
            nn.Linear(num_audio_features, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(0.3),
            nn.ReLU()
        )

        self.embedding_fc = nn.Sequential(
            nn.Linear(num_embedding_features, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(0.3),
            nn.ReLU()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=8,
            dropout=0.35,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(self.hidden_size // 2, 1)
        )

    def forward(self, text_input, audio_input, embedding_input):
        input_ids = text_input['input_ids'].squeeze(1)
        attention_mask = text_input['attention_mask'].squeeze(1)
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_embeddings = text_outputs.last_hidden_state
        audio_embeddings = self.audio_fc(audio_input)  
        audio_embeddings = audio_embeddings.unsqueeze(1) 
        embedding_embeddings = self.embedding_fc(embedding_input)  
        embedding_embeddings = embedding_embeddings.unsqueeze(1)
        combined_embeddings = torch.cat(
            (audio_embeddings, embedding_embeddings, text_embeddings),
            dim=1
        )  
        combined_embeddings = combined_embeddings.permute(1, 0, 2)
        transformer_output = self.transformer_encoder(combined_embeddings) 
        pooled_output = transformer_output[0] 
        logits = self.classifier(pooled_output).squeeze(-1)

        return logits

    def reset_parameters(self):
        def reset_layer_parameters(layer):
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.audio_fc.apply(reset_layer_parameters)
        self.embedding_fc.apply(reset_layer_parameters)
        self.transformer_encoder.apply(reset_layer_parameters)
        self.classifier.apply(reset_layer_parameters)
