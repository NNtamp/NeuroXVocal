import os

# Paths
BASE_DIR = 'C:/Users/30697/Desktop/Personal Projects/NeuroXVocal/data/ADReSSo21_audio/diagnosis/train/processed_data'  # Update this to your dataset path, take into consideration the structure below
AD_TEXT_DIR = 'C:/Users/30697/Desktop/Personal Projects/NeuroXVocal/data/ADReSSo21_audio/diagnosis/train/processed_data/ad'
CN_TEXT_DIR = 'C:/Users/30697/Desktop/Personal Projects/NeuroXVocal/data/ADReSSo21_audio/diagnosis/train/processed_data/cn'
AD_CSV = 'C:/Users/30697/Desktop/Personal Projects/NeuroXVocal/data/ADReSSo21_audio/diagnosis/train/processed_data/ad/audio_features_ad.csv'
CN_CSV = 'C:/Users/30697/Desktop/Personal Projects/NeuroXVocal/data/ADReSSo21_audio/diagnosis/train/processed_data/cn/audio_features_cn.csv'

# Model configuration
TEXT_EMBEDDING_MODEL = 'microsoft/deberta-base'
NUM_MFCC_FEATURES = 47
CUDA = True 

# Training parameters
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
K_FOLDS = 5

# Saving paths
SAVE_MODEL_PATH = 'C:/Users/30697/Desktop/Personal Projects/NeuroXVocal/results'
LOG_PATH = 'C:/Users/30697/Desktop/Personal Projects/NeuroXVocal/results'
