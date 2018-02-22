'''
This file contains all the configurations needed by the model and the training procedures.
All hyperparameters will be defined here
'''
from enums import DataLoaderType, ModelType, DataMode


#---------------------------------------------------------------------------------
DATALOADER_TYPE = DataLoaderType.CLEVR
MODEL_TYPE = ModelType.FILM
#-----------------------------------------------------------------------------------
# Dataset Paths
CLEVR_DATASET_PATH = '../vqa/datasets/CLEVR/'
FIGUREQA_DATASET_PATH = '../datasets/FIGUREQA/'
SHAPES_DATASET_PATH = '../datasets/SHAPES/'

#-------------------------------------------------------------------------------------
# MODEL PATHS
MODEL_SAVE_FILEPATH = './film_model.pt'
MODEL_LOAD_FILEPATH = ''

#----------------------------------------------------------------------------------------
# MISC Params
TRAIN_MODE = True 
USE_GPU = True
DISPLAY_LOSS_EVERY = 1
DISPLAY_METRICS_EVERY = 1
#-----------------------------------------------------------------------------------------
# TRAINING HYPERPARAMETERS
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
EPOCH_COUNT = 1
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
CHANNEL_COUNT = 'CHANNEL_COUNT'
STOP = 'STOP'
#-------------------------------------------------------------------------------------------
# Dataset Dictionaries
CLEVR_DICTIONARY = {IMAGE_SIZE : 30, QUESTION_VOCAB_SIZE : 81, ANSWER_VOCAB_SIZE : 28, MAX_QUESTION_LENGTH : 44, CHANNEL_COUNT : 3}
CLEVR_QUESTION_FILES = {DataMode.TRAIN : 'CLEVR_train_questions.json', DataMode.TEST : 'CLEVR_test_questions.json', DataMode.VAL : 'CLEVR_val_questions.json'}
FIGUREQA_DICTIONARY = {IMAGE_SIZE : 30, QUESTION_VOCAB_SIZE : 85, ANSWER_VOCAB_SIZE : 2, MAX_QUESTION_LENGTH : 12, CHANNEL_COUNT : 3}
FIGUREQA_QUESTION_FILES = {DataMode.TRAIN : 'FigureQA_train.json', DataMode.TEST : 'FigureQA_test.json', DataMode.VAL : 'FigureQA_val.json'}
SHAPES_DICTIONARY = {IMAGE_SIZE : 30, QUESTION_VOCAB_SIZE : 15, ANSWER_VOCAB_SIZE : 2, MAX_QUESTION_LENGTH : 12}
SHAPES_QUESTION_FILES = {DataMode.TRAIN : 'shapes_train.json', DataMode.TEST : 'shapes_test.json', DataMode.VAL : 'shapes_val.json'}
