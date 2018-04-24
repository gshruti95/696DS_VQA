'''
This file provides the class definition for the FigureQA Dataset
'''
import sys
import os
import torch
import torch.utils.data as data
from scipy import misc
import config
import utilities
import json
from enums import DataMode

class FigureQADataset(data.Dataset):
    '''
    Provides deifnitions for the FigureQA dataset loader.
    '''
    def __init__(self, data_mode, question_vocab, answer_vocab, USE_SAMPLING = True):
        self.question_vocab = question_vocab
        self.answer_vocab = answer_vocab
        self.images_path = config.FIGUREQA_DATASET_PATH + config.IMAGES + '/' + data_mode + '/'
        questions_path = config.FIGUREQA_DATASET_PATH + config.QUESTIONS + '/'
        question_json_filename = config.FIGUREQA_QUESTION_FILES[data_mode]
        question_json = file(questions_path + question_json_filename).read()
        questions_list = json.loads(question_json)
        self.questions_list = questions_list
        # perform pruning of the questions list
        if USE_SAMPLING == True and data_mode == DataMode.TRAIN:
            sampling_ratio = 0.1
            self.perform_question_sampling(sampling_ratio)

    def __getitem__(self, index):
        question_item = self.questions_list[index]
        question = question_item[config.QUESTION_KEY]
        answer_label = -1
        if question_item.has_key(config.ANSWER_KEY) == True:
            answer = question_item[config.ANSWER_KEY]
            answer_label = utilities.encode_answer(answer, self.answer_vocab)
        
        image_filename = question_item[config.IMAGE_FILENAME_KEY]
        encoded_question = utilities.encode_question(question, self.question_vocab, config.FIGUREQA_DICTIONARY[config.MAX_QUESTION_LENGTH])
        image_size = config.FIGUREQA_DICTIONARY[config.IMAGE_SIZE]
        image = misc.imread(self.images_path + str(image_size) + '/' + image_filename)
        modified_image = image[:, :, 0 : config.FIGUREQA_DICTIONARY[config.CHANNEL_COUNT]]
        return (modified_image / 255., encoded_question, answer_label)

    def __len__(self):
        return len(self.questions_list)
    
    def perform_question_sampling(self, sampling_ratio):
        temp_questions_list = []
        question_type_dict = {}
        for question_item in self.questions_list:
            qid = question_item['question_type']
            if question_type_dict.has_key(qid) == False:
                question_type_dict[qid] = 1
            else:
                question_type_dict[qid] += 1

        question_type_count = [0 for _ in xrange(len(question_type_dict.keys()))]

        for question_item in self.questions_list:
            qid = question_item['question_type']
            qid_current_count = question_type_count[qid]
            qid_max_count = question_type_dict[qid]
            if qid_current_count < int(qid_max_count * sampling_ratio):
                question_type_count[qid] += 1
                temp_questions_list.append(question_item)
        
        self.questions_list = temp_questions_list
