import os

# Paths
BASE_DIR = '/home/ntampakisnik/neroxvocal/processed_data'
AD_TEXT_DIR = '/home/ntampakisnik/neroxvocal/processed_data/ad/'
CN_TEXT_DIR = '/home/ntampakisnik/neroxvocal/processed_data/cn/'
AD_CSV = '/home/ntampakisnik/neroxvocal/processed_data/ad/audio_features_ad.csv'
CN_CSV = '/home/ntampakisnik/neroxvocal/processed_data/cn/audio_features_cn.csv'

# Model configuration
TEXT_EMBEDDING_MODEL = 'google/electra-base-discriminator'
NUM_MFCC_FEATURES = 47
AUDIO_CHANNELS = 1
CUDA = True

# Training parameters
BATCH_SIZE = 4
EPOCHS = 140
LEARNING_RATE = 1e-4
K_FOLDS = 8

# Saving paths
SAVE_MODEL_PATH = '/home/ntampakisnik/neroxvocal/results/model'
LOG_PATH = '/home/ntampakisnik/neroxvocal/results/training.log'


