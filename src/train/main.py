import torch
from config import *
from data_loader import create_dataloaders
from models import MultiModalDementiaClassifier
from train import train_model

def main():
    dataloaders = create_dataloaders(BATCH_SIZE, AD_TEXT_DIR, CN_TEXT_DIR, AD_CSV, CN_CSV, K_FOLDS)
    model = MultiModalDementiaClassifier(NUM_MFCC_FEATURES, TEXT_EMBEDDING_MODEL)
    device = torch.device('cuda' if torch.cuda.is_available() and CUDA else 'cpu')
    model.to(device)
    train_model(model, dataloaders, EPOCHS, LEARNING_RATE, LOG_PATH, SAVE_MODEL_PATH, device)

if __name__ == "__main__":
    main()

