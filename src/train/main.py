import torch
from config import *
from data_loader import create_dataloaders
from models import MultiModalDementiaClassifier
from train import train_model
import torch.nn as nn

def main():
    print("Starting training script...")

    with open(LOG_PATH, 'w') as f:
        pass

    dataloaders = create_dataloaders(BATCH_SIZE, AD_TEXT_DIR, CN_TEXT_DIR, AD_CSV, CN_CSV, K_FOLDS)
    model = MultiModalDementiaClassifier(num_audio_features=AUDIO_CHANNELS, text_embedding_model=TEXT_EMBEDDING_MODEL, audio_length=NUM_MFCC_FEATURES)
    device = torch.device('cuda' if torch.cuda.is_available() and CUDA else 'cpu')
    model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    else:
        print("Using a single GPU or CPU")

    train_model(model, dataloaders, EPOCHS, LEARNING_RATE, LOG_PATH, SAVE_MODEL_PATH, device)


if __name__ == "__main__":
    main()

