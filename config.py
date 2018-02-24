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
WORKING_DIR = './' # This is the path to vqa directory which also contains datasets, not VQA repo
CLEVR_DATASET_PATH = WORKING_DIR + 'datasets/CLEVR/'
FIGUREQA_DATASET_PATH = WORKING_DIR + 'datasets/FIGUREQA/'
SHAPES_DATASET_PATH = WORKING_DIR + 'datasets/SHAPES/'

#-------------------------------------------------------------------------------------
# MODEL PATHS
CHECKPOINT_FREQUENCY = 1
MODEL_SAVE_DIRNAME = 'baseline/' #please dont add any file extensions
MODEL_SAVE_FILENAME = MODEL_SAVE_DIRNAME[:-1] + '_'
MODEL_SAVE_FILEPATH = WORKING_DIR + MODEL_SAVE_DIRNAME + MODEL_SAVE_FILENAME
MODEL_LOAD_FILEPATH = MODEL_SAVE_FILEPATH + '9.pt'  # Only set this variable with appropriate model path if you need to resume training or use for inferencce

#----------------------------------------------------------------------------------------
# MISC Params
TRAIN_MODE = True
USE_GPU = True
DISPLAY_LOSS_EVERY = 20
DISPLAY_METRICS_EVERY = 5
#-----------------------------------------------------------------------------------------
# TRAINING HYPERPARAMETERS
BATCH_SIZE = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0
EPOCH_COUNT = 10
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
PYTORCH_FILE_EXTENSION = '.pt'
EPOCH_STRING = 'EPOCH_STRING' 
MODEL = 'MODEL'
#-------------------------------------------------------------------------------------------
# Dataset Dictionaries
CLEVR_DICTIONARY = {IMAGE_SIZE : 30, QUESTION_VOCAB_SIZE : 81, ANSWER_VOCAB_SIZE : 28, MAX_QUESTION_LENGTH : 44, CHANNEL_COUNT : 3}
CLEVR_QUESTION_FILES = {DataMode.TRAIN : 'clevr_train.json', DataMode.TEST : 'clevr_test.json', DataMode.VAL : 'clevr_val.json'}
FIGUREQA_DICTIONARY = {IMAGE_SIZE : 30, QUESTION_VOCAB_SIZE : 85, ANSWER_VOCAB_SIZE : 2, MAX_QUESTION_LENGTH : 12, CHANNEL_COUNT : 3}
FIGUREQA_QUESTION_FILES = {DataMode.TRAIN : 'FigureQA_train.json', DataMode.TEST : 'FigureQA_test.json', DataMode.VAL : 'FigureQA_val.json'}
SHAPES_DICTIONARY = {IMAGE_SIZE : 30, QUESTION_VOCAB_SIZE : 15, ANSWER_VOCAB_SIZE : 2, MAX_QUESTION_LENGTH : 12}
SHAPES_QUESTION_FILES = {DataMode.TRAIN : 'shapes_train.json', DataMode.TEST : 'shapes_test.json', DataMode.VAL : 'shapes_val.json'}
