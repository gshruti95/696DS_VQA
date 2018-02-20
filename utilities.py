'''
This file contains the most common utilities for metric computation
'''

import sys
import os
import config
import numpy as np


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
    # TODO
    #return confusion_matrix
    pass
