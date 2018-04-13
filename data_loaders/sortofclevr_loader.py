import sys
import os
import torch
import torch.utils.data as data
from scipy import misc
import config
import utilities
import json
from enums import DataMode
import pickle
import numpy as np

class SortOfClevrDataset(data.Dataset):
    def __init__(self, data_mode):
        dirs = config.SORT_OF_CLEVR_DATASET_PATH
        filename = os.path.join(dirs,'sort-of-clevr.pickle')
        with open(filename, 'rb') as f:
            train_datasets, test_datasets = pickle.load(f)
        self.questions_list = []
        self.answers_list = []
        self.img_list = []
        if data_mode == DataMode.TRAIN:
            dataset = train_datasets
        else:
            dataset = test_datasets
        for img, relations, norelations in dataset:
            for qst,ans in zip(relations[0], relations[1]):
                self.img_list.append(img)
                self.questions_list.append(qst)
                self.answers_list.append(ans)
            for qst,ans in zip(norelations[0], norelations[1]):
                self.img_list.append(img)
                self.questions_list.append(qst)
                self.answers_list.append(ans)

        '''
        sort_of_clevr_dictionary = config.SORT_OF_CLEVR_DICTIONARY
        self.answer_vocab = sort_of_clevr_dictionary[config.REL_ANSWER_VOCAB_SIZE]
        
        self.images_path = config.SORT_OF_CLEVR_DATASET_PATH + config.IMAGES + '/' + data_mode + '/'
        questions_path = config.SORT_OF_CLEVR_DATASET_PATH + config.QUESTIONS + '/'
        question_json_filename = config.SORT_OF_CLEVR_QUESTION_FILES[data_mode]
        question_json = file(questions_path + question_json_filename).read()
        questions_dictionary = json.loads(question_json)
        self.questions_list, self.answers_list = self.perform_question_preprocessing(questions_dictionary)
        '''
        
    
    def __getitem__(self, index):
        return (self.img_list[index], self.questions_list[index], self.answers_list[index])
        '''
        temp_question = self.questions_list[index]
        temp_question_list = temp_question.replace('[' , '').replace(']' , '').split(' ')
        encoded_question = [float(x) for x in temp_question_list if len(x) > 0]
        answer_label = int(self.answers_list[index])
        image_filename = str(index // 2) + '.jpg'
        image_size = config.SORT_OF_CLEVR_DICTIONARY[config.IMAGE_SIZE]
        modified_image = misc.imread(self.images_path + str(image_size) + '/' + image_filename)
        return (modified_image, encoded_question, answer_label)
        '''

    def __len__(self):
        return len(self.questions_list)


    def perform_question_preprocessing(self, questions_dictionary):
        questions_list = []
        answers_list = []
        for index in questions_dictionary.keys():
            question_item = questions_dictionary[index]['norel_question']
            answer_item = questions_dictionary[index]['norel_answer']
            questions_list.append(question_item)
            answers_list.append(answer_item)
            question_item = questions_dictionary[index]['rel_question']
            answer_item = questions_dictionary[index]['rel_answer']
            questions_list.append(question_item)
            answers_list.append(answer_item)
        return (questions_list, answers_list)