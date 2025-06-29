import os

TEXT_EMBEDDING_MODEL = 'microsoft/deberta-v3-base'
NUM_MFCC_FEATURES = 47
NUM_EMBEDDING_FEATURES = 768

BASE_DIR = r"/NeuroXVocal"
MODEL_PATH = os.path.join(BASE_DIR, "results/<your_trained_model>.pth")
TRAIN_DIR = os.path.join(BASE_DIR, "src/train")
EXPLAINER_DIR = os.path.join(BASE_DIR, "src/explainer")