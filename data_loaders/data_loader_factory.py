'''
This file contains the factory definitions for fetching the appropriate data loader for model training
'''
from enums import DataLoaderType, DataMode
from clevr_loader import CLEVRDataset
from figureQA_loader import FigureQADataset
from shapes_loader import ShapesDataset
import config
import utilities
import numpy as np

def get_data_loader(data_loader_type, data_mode):
    '''
    Based on the data_loader type, return the data loader
    '''
    question_vocab, answer_vocab = get_dataset_vocab(data_loader_type)
    if data_loader_type == DataLoaderType.CLEVR:
        return CLEVRDataset(data_mode, question_vocab, answer_vocab)
    elif data_loader_type == DataLoaderType.FIGUREQA:
        return FigureQADataset(data_mode, question_vocab, answer_vocab)
    elif data_loader_type == DataLoaderType.SHAPES:
        return ShapesDataset(data_mode, question_vocab, answer_vocab)


def get_dataset_vocab(data_loader_type):
    question_json_file = ''
    if data_loader_type == DataLoaderType.CLEVR:
        question_json_file = config.CLEVR_DATASET_PATH + config.QUESTIONS + '/' + config.CLEVR_QUESTION_FILES[DataMode.TRAIN]
    elif data_loader_type == DataLoaderType.FIGUREQA:
        question_json_file = config.FIGUREQA_DATASET_PATH + config.QUESTIONS + '/' + config.FIGUREQA_QUESTION_FILES[DataMode.TRAIN]
    elif data_loader_type == DataLoaderType.SHAPES:
        question_json_file = config.SHAPES_DATASET_PATH + config.QUESTIONS + '/' + config.SHAPES_QUESTION_FILES[DataMode.TRAIN]
    
    return utilities.get_vocabulary(question_json_file)


def get_dataset_dictionary(data_loader_type):
    if data_loader_type == DataLoaderType.CLEVR:
        return config.CLEVR_DICTIONARY
    elif data_loader_type == DataLoaderType.FIGUREQA:
        return config.FIGUREQA_DICTIONARY
    elif data_loader_type == DataLoaderType.SHAPES:
        return config.SHAPES_DICTIONARY


def get_dataset_specific_confusion_matrix(data_loader_type):
    class_count = 1
    if data_loader_type == DataLoaderType.CLEVR:
        class_count = config.CLEVR_DICTIONARY[config.ANSWER_VOCAB_SIZE]
    elif data_loader_type == DataLoaderType.FIGUREQA:
        class_count = config.FIGUREQA_DICTIONARY[config.ANSWER_VOCAB_SIZE]
    elif data_loader_type == DataLoaderType.SHAPES:
        class_count = config.SHAPES_DICTIONARY[config.ANSWER_VOCAB_SIZE]

    return np.zeros((class_count, class_count))




