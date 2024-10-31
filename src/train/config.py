import os

# Paths
BASE_DIR = 'path/to/processed_data/'
AD_TEXT_DIR = 'path/to/processed_data/ad/'
CN_TEXT_DIR = 'path/to/processed_data/cn/'
AD_CSV = 'path/to/processed_data/ad/audio_features_ad.csv'
CN_CSV = 'path/to/processed_data/cn/audio_features_cn.csv'
AD_EMBEDDING_CSV = 'path/to/processed_data/ad/audio_embeddings_ad.csv'
CN_EMBEDDING_CSV = 'path/to/processed_data/cn/audio_embeddings_cn.csv'

# Model configuration
TEXT_EMBEDDING_MODEL = 'microsoft/deberta-v3-base'
NUM_MFCC_FEATURES = 47
NUM_EMBEDDING_FEATURES = 768 
AUDIO_CHANNELS = 1
CUDA = True

# Training parameters
BATCH_SIZE = 'Desired number of samples per training batch'
EPOCHS = 'Total number of training epochs'
LEARNING_RATE = 'Learning rate for the optimizer'
WEIGHT_DECAY = 'Weight decay (L2 regularization) rate'
NUM_FOLDS = 'Number of folds for cross-validation'
SAVE_BEST_MODEL = 'Flag to save only the best-performing model (True/False)'

# Early stopping criteria
EARLY_STOPPING_PATIENCE ='Number of epochs with no improvement to trigger early stopping'

# Saving paths
SAVE_MODEL_PATH = 'path/to/results/folder/model' #Create a folder "results" for saving the model
LOG_PATH = 'path/to/results/folder/training.log' #Create a folder "results" for saving logs


