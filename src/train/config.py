import os

# Paths
BASE_DIR = '/home/ntampakisnik/neroxvocal/processed_data/'
AD_TEXT_DIR = '/home/ntampakisnik/neroxvocal/processed_data/ad/'
CN_TEXT_DIR = '/home/ntampakisnik/neroxvocal/processed_data/cn/'
AD_CSV = '/home/ntampakisnik/neroxvocal/processed_data/ad/audio_features_ad.csv'
CN_CSV = '/home/ntampakisnik/neroxvocal/processed_data/cn/audio_features_cn.csv'
AD_EMBEDDING_CSV = '/home/ntampakisnik/neroxvocal/processed_data/ad/audio_embeddings_ad.csv'
CN_EMBEDDING_CSV = '/home/ntampakisnik/neroxvocal/processed_data/cn/audio_embeddings_cn.csv'

# Model configuration
TEXT_EMBEDDING_MODEL = 'microsoft/deberta-v3-base'
NUM_MFCC_FEATURES = 47
NUM_EMBEDDING_FEATURES = 768 
AUDIO_CHANNELS = 1
CUDA = True

# Training parameters
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4 
NUM_FOLDS = 6  
SAVE_BEST_MODEL = False 

# Early stopping parameter
EARLY_STOPPING_PATIENCE = 5 

# Saving paths
SAVE_MODEL_PATH = '/home/ntampakisnik/neroxvocal/results/model'
LOG_PATH = '/home/ntampakisnik/neroxvocal/results/training.log'


