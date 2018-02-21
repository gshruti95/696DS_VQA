'''
This file contains all the configurations needed by the model and the training procedures.
All hyperparameters will be defined here
'''
from enums import DataLoaderType, ModelType, DataMode


#---------------------------------------------------------------------------------
DATALOADER_TYPE = DataLoaderType.SHAPES
MODEL_TYPE = ModelType.BASELINE
#-----------------------------------------------------------------------------------
# Dataset Paths
CLEVR_DATASET_PATH = '../../datasets/CLEVR/'
FIGUREQA_DATASET_PATH = '../../datasets/FIGUREQA/'
SHAPES_DATASET_PATH = '../../datasets/SHAPES/'

#-------------------------------------------------------------------------------------
# MODEL PATHS
MODEL_SAVE_FILEPATH = './baseline_model.pt'
MODEL_LOAD_FILEPATH = ''

#----------------------------------------------------------------------------------------
# MISC Params
TRAIN_MODE = True 
USE_GPU = False
DISPLAY_LOSS_EVERY = 1
DISPLAY_METRICS_EVERY = 1
#-----------------------------------------------------------------------------------------
# TRAINING HYPERPARAMETERS
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0
EPOCH_COUNT = 5
#------------------------------------------------------------------------------------------
# KEYWORDS
IMAGES = 'images'
QUESTIONS = 'questions'
ANSWER_KEY = 'answer'
QUESTION_KEY = 'question'
IMAGE_FILENAME_KEY = 'image_filename'
IMAGE_SIZE = 'IMAGE_SIZE'
QUESTION_VOCAB_SIZE = 'QUESTION_VOCAB_SIZE'
ANSWER_VOCAB_SIZE = 'ANSWER_VOCAB_SIZE'
MAX_QUESTION_LENGTH = 'MAX_QUESTION_LENGTH'
STOP = 'STOP'
#-------------------------------------------------------------------------------------------
# Dataset Dictionaries
CLEVR_DICTIONARY = {IMAGE_SIZE : 32, QUESTION_VOCAB_SIZE : 80, ANSWER_VOCAB_SIZE : 28, MAX_QUESTION_LENGTH : 44}
CLEVR_QUESTION_FILES = {DataMode.TRAIN : 'clevr_train.json', DataMode.TEST : 'clevr_test.json', DataMode.VAL : 'clevr_val.json'}
FIGUREQA_DICTIONARY = {IMAGE_SIZE : 32, QUESTION_VOCAB_SIZE : 84, ANSWER_VOCAB_SIZE : 2, MAX_QUESTION_LENGTH : 12}
FIGUREQA_QUESTION_FILES = {DataMode.TRAIN : 'FigureQA_train.json', DataMode.TEST : 'FigureQA_test.json', DataMode.VAL : 'FigureQA_val.json'}
SHAPES_DICTIONARY = {IMAGE_SIZE : 30, QUESTION_VOCAB_SIZE : 14, ANSWER_VOCAB_SIZE : 2, MAX_QUESTION_LENGTH : 12}
SHAPES_QUESTION_FILES = {DataMode.TRAIN : 'shapes_train.json', DataMode.TEST : 'shapes_test.json', DataMode.VAL : 'shapes_val.json'}