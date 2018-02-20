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


def check_and_get_gpu_instance(item):
    if torch.cuda.is_available() == True and config.USE_GPU == True:
        return item.cuda()
    return item

def train(model, data_loader, optimizer, criterion, epoch_count):
    # train the model for fixed number of epochs
    # print the training loss , train accuracy and validation accuracy
    model = check_and_get_gpu_instance(model)
    for epoch_index in xrange(epoch_count):
        for mini_index, (images, questions, labels) in enumerate(data_loader):
            # convert the images, questions and the labels to variables and then to cuda instances
            images = Variable(images, requires_grad = False)
            questions = Variable(questions, requires_grad = False)
            labels = Variable(torch.stack(labels, dim = 1), requires_grad = False)
            labels = labels.permute(1, 0, 2)
            images = check_and_get_gpu_instance(images)
            questions = check_and_get_gpu_instance(questions)
            target_labels = check_and_get_gpu_instance(labels)
            target_labels = target_labels.float()
            # forward, backward, step
            model.zero_grad()
            images = images.permute(0, 3, 1, 2)
            predictions = model(images.float(), questions.float())
            #print(predictions.view(int(predictions.size()[0] * predictions.size()[1]), -1).size())
            loss = criterion(predictions.view(int(predictions.size()[0] * predictions.size()[1]), -1) , target_labels)
            if mini_index % 50 == 0:
            	print('loss for %d/%d epoch, %d/%d batch = %f' % (epoch_index + 1, epoch_count, mini_index + 1, len(data_loader), loss.data[0]))
            loss.backward()
            optimizer.step()
             
        if (epoch_index + 1) % 1 == 0:
            predict(model, config.DataMode.TRAIN)
            predict(model, config.DataMode.VAL)
    
    return model

def fit(model):
    # get the data loader iterator
    data_loader = get_data(config.DataMode.TRAIN)
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
    confusion_matrix = np.zeros((config.CLASS_COUNT - 2, config.CLASS_COUNT - 2)) #(True class, predicted class)
    for i, (images, labels) in enumerate(data_loader):
        images = Variable(images, requires_grad = False)
        labels = Variable(torch.stack(labels, dim = 1), requires_grad = False)
        labels = labels.permute(1, 0, 2)
        images = check_and_get_gpu_instance(images)
        input_labels = check_and_get_gpu_instance(labels[:, : -1, :])
        target_labels = check_and_get_gpu_instance(labels[:, 1 :, :])
        target_labels = target_labels.float()
        _, target_labels = torch.max(target_labels, 2) #(N, T)
        images = images.permute(0, 3, 1, 2)
        predictions = model(images.float(), input_labels.float())
        _, predicted_array = torch.max(predictions, 2) #(N, T)
        predicted_array = predicted_array.cpu().data.numpy() #(N, T)
        target_array = target_labels.cpu().data.numpy() #(N, T)
        confusion_matrix = confusion_matrix + utilities.get_confusion_matrix(predicted_array, target_array)
        
        if print_values:
            for index in xrange(predicted_array.shape[0]):
                true_string = utilities.decode_string(target_array[index])
                predicted_string = utilities.decode_string(predicted_array[index])
                print('True string = ' + true_string + ' and predicted string = ' + predicted_string)

    # compute accuracy, precision, recall and f1-score
    print('Accuracy = ' + str(utilities.get_accuracy(confusion_matrix)))
    print('Precision = ' + str(utilities.get_precision(confusion_matrix)))
    print('Recall = ' + str(utilities.get_recall(confusion_matrix)))
    print('F1_score = ' + str(utilities.get_f1_score(confusion_matrix)))


def load_model(model_path):
    try:
        model = torch.load(model_path)
        return model
    except:
        print('Unable to load the model')

def save_model(model, model_path):
    try:
        torch.save(model, model_path)
    except:
        print('Unable to save the model')

def get_data(data_mode):
    '''
    Returns the data loader iterator based on the data_mode(train, val or test)
    '''
    custom_dataset = data_loader_factory.get_data_loader('ClevrDataset', data_mode)
    data_loader = torch.utils.data.DataLoader(dataset = custom_dataset, batch_size = config.BATCH_SIZE, shuffle = True)
    return data_loader


def main():
    model = None
    if config.TRAIN_MODE == True:
        model = net_factory.get_network('baseline')
        model = fit(model)
        save_model(model, config.MODEL_SAVE_FILEPATH)
    # set the model in evaluation mode
    else:
        model = load_model(config.MODEL_LOAD_FILEPATH)    
    # perform prediction
    #predict(model, config.DataMode.TEST, print_values = True)


if __name__ == '__main__':
    main()
