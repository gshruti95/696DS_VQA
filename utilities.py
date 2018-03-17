'''
This file contains the most common utilities for metric computation
'''

import sys
import os
import config
import numpy as np
import json
from data_loaders import data_loader_factory
from scipy import misc
from enums import DataMode


def perform_dataset_preprocessing():
    data_modes = [DataMode.TRAIN, DataMode.VAL, DataMode.TEST]
    
    for mode in data_modes:
        dataset_path = data_loader_factory.get_dataset_path(config.DATALOADER_TYPE) + config.IMAGES + '/' + mode + '/'
        if os.path.exists(dataset_path) == False:
            Exception('Dataset path does not exist')
        image_files = os.listdir(dataset_path)
        dataset_dict = data_loader_factory.get_dataset_dictionary(config.DATALOADER_TYPE)
        image_size = dataset_dict[config.IMAGE_SIZE]
        # Check if a particular folder exists for the desired size
        temp_dataset_path = dataset_path + str(image_size) + '/'
        # If the desired folder exists, avoiding repeated pre-processing
        if os.path.exists(temp_dataset_path) == True:
            return
        # Perform preprocessing
        os.mkdir(temp_dataset_path)
        for filename in image_files:
            filepath = dataset_path + filename
            if os.path.isdir(filepath) == True:
                continue
            image = misc.imread(filepath)
            modified_image = misc.imresize(image, (image_size, image_size))
            misc.imsave(temp_dataset_path + filename, modified_image)


def get_vocabulary(json_filepath):
    json_data = file(json_filepath).read()
    question_list = json.loads(json_data)
    word_dictionary = {}
    answer_dictionary = {}
    for item in question_list:
        question = item[config.QUESTION_KEY]
        answer = item[config.ANSWER_KEY]
        word_list = question.replace('?' , '').replace(';' , '').lower().split(' ')
        for word in word_list:
            if word == '':
                continue
            word_dictionary[word] = True
        if answer == '':
            continue
        answer_dictionary[answer] = True
        
    return (word_dictionary.keys(), answer_dictionary.keys())

def encode_question(question_string, question_vocab, max_question_length):
    encoded_list = []
    word_list = question_string.replace('?','').replace(';','').replace('\n','').lower().split(' ')
    counter = 0
    for word in word_list:
        if len(word) == 0:
            continue
        index = question_vocab.index(word)
        encoded_list.append(index)
        counter = counter + 1
    
    stop_symbol_index = len(question_vocab)
    while counter < max_question_length:
        encoded_list.append(stop_symbol_index)
        counter = counter + 1

    return encoded_list


def encode_answer(answer_word, answer_vocab):
    return answer_vocab.index(answer_word)

def get_accuracy(confusion_matrix):
    match_count = 0
    for index in xrange(confusion_matrix.shape[0]):
        match_count += confusion_matrix[index, index]
    total_count = np.sum(np.sum(confusion_matrix))
    if total_count == 0:
        return 0.0
    return (match_count * 1.0) / (total_count * 1.0)
    
def get_precision(confusion_matrix):
    mean_avg_precision = 0
    denominator_matrix = np.sum(confusion_matrix, axis=0)
    denominator_count = 0
    for index in xrange(confusion_matrix.shape[0]):
        if denominator_matrix[index] == 0:
            continue
        mean_avg_precision += (confusion_matrix[index, index] * 1.0) / denominator_matrix[index]
        denominator_count += 1
    
    try:
        return (mean_avg_precision * 1.0) / denominator_count
    except:
        return 0.0

def get_recall(confusion_matrix):
    mean_avg_recall = 0
    denominator_matrix = np.sum(confusion_matrix, axis=1)
    denominator_count = 0
    for index in xrange(confusion_matrix.shape[0]):
        if denominator_matrix[index] == 0:
            continue
        mean_avg_recall += (confusion_matrix[index, index] * 1.0) / denominator_matrix[index]
        denominator_count += 1
    try:
        return (mean_avg_recall * 1.0) / denominator_count
    except:
        return 0.0


def get_f1_score(confusion_matrix):
    precision = get_precision(confusion_matrix)
    recall = get_recall(confusion_matrix)
    if (precision + recall) == 0.0:
        return 0.0
    try:
        f1 = (2.0 * precision * recall) / (precision + recall)
        return f1
    except:
        return 0.0

def get_confusion_matrix(predicted_array, true_array):
    temp_confusion_matrix = data_loader_factory.get_dataset_specific_confusion_matrix(config.DATALOADER_TYPE)
    for index in xrange(true_array.shape[0]):
        true_class = true_array[index]
        predicted_class = predicted_array[index]
        temp_confusion_matrix[true_class, predicted_class] += 1
return temp_confusion_matrix
