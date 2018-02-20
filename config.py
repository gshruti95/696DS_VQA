'''
This file contains all the configurations needed by the model and the training procedures.
All hyperparameters will be defined here
'''

class DataMode(enumerate):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

TARGET_FILEPATH = './data/'

# TRAINING HYPERPARAMETERS
TRAIN_MODE = False
MODEL_SAVE_FILEPATH = './trained_model_15_25_0001_m40.pt'
MODEL_LOAD_FILEPATH = '../models/dotfeatures_attention.pt'
USE_GPU = False
DISPLAY_LOSS_EVERY = 1
DISPLAY_METRICS_EVERY = 1

# MODEL HYPERPARAMETERS
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0
EPOCH_COUNT = 5
