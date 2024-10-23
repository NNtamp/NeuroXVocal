import torch
import torch.nn as nn
from transformers import DebertaModel

class MultiModalDementiaClassifier(nn.Module):
    def __init__(self, num_audio_features, text_embedding_model):
        super(MultiModalDementiaClassifier, self).__init__()
        self.text_model = DebertaModel.from_pretrained(text_embedding_model)
        self.text_fc = nn.Linear(self.text_model.config.hidden_size, 256)
        self.audio_fc = nn.Linear(num_audio_features, 256)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, text_input, audio_input):
        input_ids = text_input['input_ids']
        attention_mask = text_input['attention_mask']

        text_embeddings = self.text_model(input_ids, attention_mask=attention_mask)['last_hidden_state']
        text_features = self.text_fc(text_embeddings[:, 0, :])
        
        audio_features = self.audio_fc(audio_input)
        
        combined = torch.cat((text_features, audio_features), dim=1)
        x = torch.relu(self.fc1(combined))
        output = torch.sigmoid(self.fc2(x))
        
        return output



