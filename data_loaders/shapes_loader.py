
import sys
import os
import torch
import torch.utils.data as data
from scipy import misc
import config
import utilities
import json

class ShapesDataset(data.Dataset):
    '''
    Provides deifnitions for the SHAPES dataset loader.
    '''
    def __init__(self, data_mode, question_vocab, answer_vocab):
        self.question_vocab = question_vocab
        self.answer_vocab = answer_vocab
        self.images_path = config.SHAPES_DATASET_PATH + config.IMAGES + '/' + data_mode + '/'
        questions_path = config.SHAPES_DATASET_PATH + config.QUESTIONS + '/'
        question_json_filename = config.SHAPES_QUESTION_FILES[data_mode]
        question_json = file(questions_path + question_json_filename).read()
        questions_list = json.loads(question_json)
        self.questions_list = self.perform_question_preprocessing(questions_list)
        
    
    def __getitem__(self, index):
        question_item = self.questions_list[index]
        question = question_item[config.QUESTION_KEY]
        answer_label = -1
        if question_item.has_key(config.ANSWER_KEY) == True:
            answer = question_item[config.ANSWER_KEY]
            answer_label = utilities.encode_answer(answer, self.answer_vocab)
        
        image_filename = question_item[config.IMAGE_FILENAME_KEY]
        encoded_question = utilities.encode_question(question, self.question_vocab, config.SHAPES_DICTIONARY[config.MAX_QUESTION_LENGTH])
        image_size = config.SHAPES_DICTIONARY[config.IMAGE_SIZE]
        modified_image = misc.imread(self.images_path + str(image_size) + '/' + image_filename)
        return (modified_image / 255., encoded_question, answer_label)

    def __len__(self):
        return len(self.questions_list)


    def perform_question_preprocessing(self, questions_list):
        modified_list = []
        for item in questions_list:
            if item.has_key(config.ANSWER_KEY) == True:
                answer = item[config.ANSWER_KEY]
                if len(answer) == 0:
                    continue
            modified_list.append(item)
        return modified_list