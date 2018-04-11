'''
This file contains all the configurations needed by the model and the training procedures.
All hyperparameters will be defined here
'''
from enums import DataLoaderType, ModelType, DataMode


#---------------------------------------------------------------------------------
DATALOADER_TYPE = DataLoaderType.SHAPES
MODEL_TYPE = ModelType.RELATION_NETWORK
#-----------------------------------------------------------------------------------
# Dataset Paths
WORKING_DIR = '../' # This is the path to vqa directory which contains datasets and VQA repository
CLEVR_DATASET_PATH = WORKING_DIR + 'datasets/CLEVR/'
FIGUREQA_DATASET_PATH = WORKING_DIR + 'datasets/FIGUREQA/'
SHAPES_DATASET_PATH = WORKING_DIR + 'datasets/SHAPES/'

#----------------------------------------------------------------------------------------
# MODEL PATHS
MODEL_SAVE_FILENAME = MODEL_TYPE # Default filename value set to model type specified above
MODEL_SAVE_DIRNAME = MODEL_TYPE  # Default folder value set to model type specified above
MODEL_SAVE_FILEPATH = WORKING_DIR + MODEL_SAVE_DIRNAME + '/' + MODEL_SAVE_FILENAME + '_' # Do not EDIT this variable
MODEL_LOAD_FILEPATH = '' # Provide the Relative or absolute path to the model that you wish to load for inference or to resume training

HYPERPARAM_FILENAME = 'hyperparams.txt' 

#----------------------------------------------------------------------------------------
# MISC Params
TRAIN_MODE = True
USE_GPU = True
DISPLAY_LOSS_EVERY = 20
DISPLAY_METRICS_EVERY = 1
#-----------------------------------------------------------------------------------------
# TRAINING HYPERPARAMETERS
CHECKPOINT_FREQUENCY = 50
BATCH_SIZE = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0
EPOCH_COUNT = 100
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
EPOCH_STRING = 'EPOCH' 
MODEL_STRING = 'MODEL'
LEARNING_RATE_STRING = 'LEARNING_RATE'
BATCH_SIZE_STRING = 'BATCH_SIZE'
WEIGHT_DECAY_STRING = 'WEIGHT_DECAY'
DATASET_STRING = 'DATASET'
ARCHITECTURE_STRING = 'ARCHITECTURE'
MODEL_FILENAME_PREFIX_STRING = 'MODEL_FILENAME_PREFIX'
QUESTION_EMBEDDING_SIZE = 'QUESTION_EMBEDDING_SIZE'
ENABLE_BATCHNORM = 'ENABLE_BATCHNORM'
LSTM_INPUT_EMBEDDING_DIM = 'LSTM_INPUT_EMBEDDING_DIM'
LSTM_HIDDEN_DIM = 'LSTM_HIDDEN_DIM'
LSTM_LAYERS = 'LSTM_LAYERS'
F_LAYER_DIM = 'F layer dimension in RN'
G_LAYER_DIM = 'G layer dimension in RN'
#-------------------------------------------------------------------------------------------
# Dataset Dictionaries
CLEVR_DICTIONARY = {IMAGE_SIZE : 30, QUESTION_VOCAB_SIZE : 81, ANSWER_VOCAB_SIZE : 28, MAX_QUESTION_LENGTH : 44, CHANNEL_COUNT : 3}
CLEVR_QUESTION_FILES = {DataMode.TRAIN : 'clevr_train.json', DataMode.TEST : 'clevr_test.json', DataMode.VAL : 'clevr_val.json'}
FIGUREQA_DICTIONARY = {IMAGE_SIZE : 30, QUESTION_VOCAB_SIZE : 85, ANSWER_VOCAB_SIZE : 2, MAX_QUESTION_LENGTH : 12, CHANNEL_COUNT : 3}
FIGUREQA_QUESTION_FILES = {DataMode.TRAIN : 'FigureQA_train.json', DataMode.TEST : 'FigureQA_test.json', DataMode.VAL : 'FigureQA_val.json'}
SHAPES_DICTIONARY = {IMAGE_SIZE : 75, QUESTION_VOCAB_SIZE : 15, ANSWER_VOCAB_SIZE : 2, MAX_QUESTION_LENGTH : 12, CHANNEL_COUNT : 3}
SHAPES_QUESTION_FILES = {DataMode.TRAIN : 'shapes_train.json', DataMode.TEST : 'shapes_test.json', DataMode.VAL : 'shapes_val.json'}

#--------------------------------------------------------------------------------------------------------

# Architecture Dictionaries
RELATION_NETWORK_DICTIONARY = {
                                QUESTION_EMBEDDING_SIZE : 256, ENABLE_BATCHNORM : False, LSTM_HIDDEN_DIM : 256, LSTM_LAYERS : 1,
                                F_LAYER_DIM : 256, G_LAYER_DIM : 256
                            }