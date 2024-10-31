import torch
from config import *
from data_loader import create_full_dataset
from models import NeuroXVocal
from train import train_model
import torch.nn as nn

def main():
    print("Starting k-fold cross-validation training script...")
    with open(LOG_PATH, 'w') as f:
        pass 

    full_dataset = create_full_dataset(
        AD_TEXT_DIR,
        CN_TEXT_DIR,
        AD_CSV,
        CN_CSV,
        AD_EMBEDDING_CSV,
        CN_EMBEDDING_CSV
    )

    device = torch.device('cuda' if torch.cuda.is_available() and CUDA else 'cpu')

    model = NeuroXVocal(
        num_audio_features=NUM_MFCC_FEATURES,
        num_embedding_features=NUM_EMBEDDING_FEATURES,
        text_embedding_model=TEXT_EMBEDDING_MODEL
    )

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    else:
        print("Using a single GPU or CPU")

    model.to(device)

    train_model(
        model,
        full_dataset,
        EPOCHS,
        LEARNING_RATE,
        LOG_PATH,
        SAVE_MODEL_PATH,
        device,
        NUM_FOLDS,
        SAVE_BEST_MODEL
    )

if __name__ == "__main__":
    main()
