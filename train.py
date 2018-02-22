'''
This file contains the following logic:
loading the training and validation set
loading the neural net architecture
defining the objective and the optimizers
perform the training
Displaying the loss during the training
the stats accuracy, precision, recall and f1 score
Saving the trained model to disk for inference
'''

import torch
import sys
import os
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import config
import utilities
import numpy as np
from architectures import net_factory
from data_loaders import data_loader_factory
from enums import DataMode


def check_and_get_gpu_instance(item):
    if torch.cuda.is_available() == True and config.USE_GPU == True:
        return item.cuda()
    return item

def train(model, data_loader, optimizer, criterion, epoch_count):
    # train the model for fixed number of epochs
    model = check_and_get_gpu_instance(model)
    for epoch_index in xrange(epoch_count):
        for mini_index, (images, questions, labels) in enumerate(data_loader):
            # convert the images, questions and the labels to variables and then to cuda instances
            images = Variable(images, requires_grad = False)
            questions = Variable(torch.stack(questions, dim = 1), requires_grad = False)
            labels = Variable(labels, requires_grad = False)
            images = check_and_get_gpu_instance(images.float())
            questions = check_and_get_gpu_instance(questions)
            target_labels = check_and_get_gpu_instance(labels)
            # forward, backward, step
            model.zero_grad()
            images = images.permute(0, 3, 1, 2)
            #print(images)
            predictions = model(images, questions)
            loss = criterion(predictions , target_labels)
            if mini_index % config.DISPLAY_LOSS_EVERY == 0:
            	print('loss for %d/%d epoch, %d/%d batch = %f' % (epoch_index + 1, epoch_count, mini_index + 1, len(data_loader), loss.data[0]))
            loss.backward()
            optimizer.step()
             
        if (epoch_index + 1) % config.DISPLAY_METRICS_EVERY == 0:
            predict(model, config.DataMode.TRAIN)
            predict(model, config.DataMode.VAL)
    
    return model

def fit(model):
    # get the data loader iterator
    data_loader = get_data(DataMode.TRAIN)
    # define the objective
    criterion = nn.CrossEntropyLoss()
    # define the optimizer
    optimizer = optim.Adam(model.parameters(), lr = config.LEARNING_RATE, weight_decay = config.WEIGHT_DECAY)
    # train
    return train(model, data_loader, optimizer, criterion, config.EPOCH_COUNT)


def predict(model, data_mode, print_values = False):
    print('Computing metrics for ' + data_mode + ' mode.')
    data_loader = get_data(data_mode)
    model = model.eval()
    confusion_matrix = data_loader_factory.get_dataset_specific_confusion_matrix(config.DATALOADER_TYPE) #(True class, predicted class)
    for _, (images, questions, labels) in enumerate(data_loader):
        images = Variable(images, requires_grad = False)
        questions = Variable(torch.stack(questions, dim = 1), requires_grad = False)
        labels = Variable(labels, requires_grad = False)
        images = check_and_get_gpu_instance(images)
        questions = check_and_get_gpu_instance(questions)
        target_labels = check_and_get_gpu_instance(labels)
        images = images.permute(0, 3, 1, 2)
        predictions = model(images.float(), questions)
        _, predicted_array = torch.max(predictions, 1) #(N,)
        predicted_array = predicted_array.cpu().data.numpy() #(N,)
        target_array = target_labels.cpu().data.numpy() #(N,)
        confusion_matrix = confusion_matrix + utilities.get_confusion_matrix(predicted_array, target_array)
        
        if print_values:
            for index in xrange(predicted_array.shape[0]):
                predicted_value = predicted_array[index]
                true_value = target_array[index]
                print('True value = ' + str(true_value) + ' and predicted value = ' + str(predicted_value))

    # compute accuracy, precision, recall and f1-score
    if data_mode == DataMode.TEST:
        return # no stats to show for test mode since ground truth is not available
    
    print('Accuracy = ' + str(utilities.get_accuracy(confusion_matrix)))
    print('Precision = ' + str(utilities.get_precision(confusion_matrix)))
    print('Recall = ' + str(utilities.get_recall(confusion_matrix)))
    print('F1_score = ' + str(utilities.get_f1_score(confusion_matrix)))



def load_model(model_path):
    '''
    Load model from the specified path
    '''
    model = torch.load(model_path)
    return model


def save_model(model, model_path):
    '''
    Save the pytorch model
    '''
    torch.save(model, model_path)


def get_data(data_mode):
    '''
    Returns the data loader iterator based on the data_mode(train, val or test)
    '''
    custom_dataset = data_loader_factory.get_data_loader(config.DATALOADER_TYPE, data_mode)
    if data_mode == DataMode.TEST:
        data_loader = torch.utils.data.DataLoader(dataset = custom_dataset, batch_size = config.BATCH_SIZE, shuffle = False)
    else:
        data_loader = torch.utils.data.DataLoader(dataset = custom_dataset, batch_size = config.BATCH_SIZE, shuffle = True)
    return data_loader


def main():
    model = None
    if config.TRAIN_MODE == True:
        model = net_factory.get_network(config.MODEL_TYPE)
        model = fit(model)
        save_model(model, config.MODEL_SAVE_FILEPATH)
    # set the model in evaluation mode
    else:
        model = load_model(config.MODEL_LOAD_FILEPATH)    
    # perform prediction
    predict(model, DataMode.TEST, print_values = True)


if __name__ == '__main__':
    main()
