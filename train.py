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
from enums import DataLoaderType

import time


def check_and_get_gpu_instance(item):
    item.cuda()
    if torch.cuda.is_available() == True and config.USE_GPU == True:
        return item.cuda()
    return item

def train(model, data_loader, optimizer, criterion, epoch_count, min_epoch_count = 0):
    predict(model, config.DataMode.TRAIN)
    predict(model, config.DataMode.VAL)
    # train the model for fixed number of epochs
    model = check_and_get_gpu_instance(model)
    for epoch_index in range(min_epoch_count, epoch_count):
        start_time = time.time()
        model.train()
        for mini_index, (images, questions, labels) in enumerate(data_loader):
            # convert the images, questions and the labels to variables and then to cuda instances
            images = Variable(images, requires_grad = False)
            questions = Variable(torch.stack(questions, dim = 1), requires_grad = False)
            #questions = questions.type(torch.FloatTensor)
            # Reducing the question length for avoiding no-op recurrent time steps in processing question through RNN
            labels = Variable(labels, requires_grad = False)
            images = check_and_get_gpu_instance(images.float())
            if config.DATALOADER_TYPE == DataLoaderType.SORT_OF_CLEVR:
                questions = check_and_get_gpu_instance(questions)
            target_labels = check_and_get_gpu_instance(labels)
            if images.size()[0] != config.BATCH_SIZE:
                continue
            # forward, backward, step
            model.zero_grad()
            images = images.permute(0, 3, 1, 2)
            predictions = model(images, questions)
            loss = criterion(predictions , target_labels)
            if mini_index % config.DISPLAY_LOSS_EVERY == 0:
            	print('loss for %d/%d epoch, %d/%d batch = %f' % (epoch_index + 1, epoch_count, mini_index + 1, len(data_loader), loss.data[0]))
            loss.backward()
            optimizer.step()
             
        if (epoch_index + 1) % config.DISPLAY_METRICS_EVERY == 0:
            predict(model, config.DataMode.TRAIN)
            predict(model, config.DataMode.VAL)
        
        if (epoch_index + 1) % config.CHECKPOINT_FREQUENCY == 0:
            model_path = config.MODEL_SAVE_FILEPATH + str(epoch_index + 1) + config.PYTORCH_FILE_EXTENSION
            save_model(model, epoch_index, model_path)
        
        sys.stdout.flush()
        print('Time taken to train epoch =' + str(time.time() - start_time))
    
    return model

def fit(model, min_epoch_count = 0):
    # get the data loader iterator
    data_loader = get_data(DataMode.TRAIN)
    # define the objective
    criterion = nn.NLLLoss() #nn.CrossEntropyLoss()
    # define the optimizer
    optimizer = optim.Adam(model.parameters(), lr = config.LEARNING_RATE, weight_decay = config.WEIGHT_DECAY)
    get_hyperparams()
    # train
    return train(model, data_loader, optimizer, criterion, config.EPOCH_COUNT, min_epoch_count = min_epoch_count)

def predict(model, data_mode, print_values = False):
    print('Computing metrics for ' + data_mode + ' mode.')
    data_loader = get_data(data_mode)
    model = model.eval()
    global_loss = 0
    batch_count = 0
    MAX_BATCH_COUNT = 30
    criterion = nn.NLLLoss()
    confusion_matrix = data_loader_factory.get_dataset_specific_confusion_matrix(config.DATALOADER_TYPE) #(True class, predicted class)
    for _, (images, questions, labels) in enumerate(data_loader):
        if batch_count > MAX_BATCH_COUNT:
            break
        images = Variable(images, requires_grad = False)
        if images.size()[0] != config.BATCH_SIZE:
            continue
        questions = Variable(torch.stack(questions, dim = 1), requires_grad = False)
        #questions = questions.type(torch.FloatTensor)
        labels = Variable(labels, requires_grad = False)
        images = check_and_get_gpu_instance(images)
        questions = check_and_get_gpu_instance(questions)
        target_labels = check_and_get_gpu_instance(labels)
        images = images.permute(0, 3, 1, 2)
        predictions = model(images.float(), questions)
        global_loss += criterion(predictions , target_labels).data[0]
        batch_count += 1
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
    print('Loss = ' + str(global_loss * 1.0 / (batch_count + 1e-10)))
    print('Accuracy = ' + str(utilities.get_accuracy(confusion_matrix)))
    print('Precision = ' + str(utilities.get_precision(confusion_matrix)))
    print('Recall = ' + str(utilities.get_recall(confusion_matrix)))
    print('F1_score = ' + str(utilities.get_f1_score(confusion_matrix)))



def load_model(model_path):
    '''
    Load model from the specified path
    '''
    state_dict = torch.load(model_path)
    model = state_dict[config.MODEL_STRING]
    return model, state_dict[config.EPOCH_STRING] + 1


def save_model(model, epoch_index, model_path):
    '''
    Save the pytorch model
    '''
    state_dict = get_hyperparams(write_to_file = False)
    state_dict[config.EPOCH_STRING] = epoch_index
    state_dict[config.MODEL_STRING] = model
    if not os.path.exists(config.WORKING_DIR + config.MODEL_SAVE_DIRNAME):
        os.makedirs(config.WORKING_DIR + config.MODEL_SAVE_DIRNAME)
    torch.save(state_dict, model_path)


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

def get_hyperparams(write_to_file = True):
    state_dict = {}
    state_dict[config.LEARNING_RATE_STRING] = config.LEARNING_RATE
    state_dict[config.BATCH_SIZE_STRING] = config.BATCH_SIZE
    state_dict[config.WEIGHT_DECAY_STRING] = config.WEIGHT_DECAY
    state_dict[config.DATASET_STRING] = config.DATALOADER_TYPE
    state_dict[config.ARCHITECTURE_STRING] = config.MODEL_TYPE
    dataset_dict = data_loader_factory.get_dataset_dictionary(config.DATALOADER_TYPE)
    state_dict[config.IMAGE_SIZE] = dataset_dict[config.IMAGE_SIZE]
    state_dict[config.CHANNEL_COUNT] = dataset_dict[config.CHANNEL_COUNT]
    if write_to_file == False:
        return state_dict
    print('MODEL HYPERPARAMETERS =' + str(state_dict))
    hyperparam_filepath = config.WORKING_DIR + config.MODEL_SAVE_DIRNAME + '/' + config.HYPERPARAM_FILENAME
    if os.path.exists(hyperparam_filepath) == False:
        if os.path.exists(config.WORKING_DIR + config.MODEL_SAVE_DIRNAME) == False:
            os.mkdir(config.WORKING_DIR + config.MODEL_SAVE_DIRNAME)
        file_object = open(hyperparam_filepath, 'w+')
    else:
        file_object = open(hyperparam_filepath, 'a+')
    state_dict[config.MODEL_FILENAME_PREFIX_STRING] = config.MODEL_SAVE_FILENAME
    file_object.writelines(str(state_dict))
    file_object.close()


def main():
    utilities.perform_dataset_preprocessing()
    model = None
    if config.TRAIN_MODE == True:
        model = net_factory.get_network(config.MODEL_TYPE)
        if len(config.MODEL_LOAD_FILEPATH) > 0:
            model, min_epoch_count = load_model(config.MODEL_LOAD_FILEPATH)
            model = fit(model, min_epoch_count)
        else:
            model = fit(model)
    # set the model in evaluation mode
    else:
        try:
            model, _ = load_model(config.MODEL_LOAD_FILEPATH)
        except:
            print('Provide appropriate model path and rerun for inference')
            return    
    # perform prediction
	predict(model, DataMode.TEST, print_values = True)


if __name__ == '__main__':
    main()