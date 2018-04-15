# this script would run multiple configurations for the hyperparameters specified and log all the reports for a specified model


import sys
import os
import config
import train
from enums import DataLoaderType
from enums import ModelType




#######################################################################################################################
dataset_list = [DataLoaderType.SHAPES, DataLoaderType.SORT_OF_CLEVR]
network_list = [ModelType.RELATION_NETWORK]
#network_list = [ModelType.RELATION_GROUP_ATTENTION_STANDARD, ModelType.RELATION_GROUP_ATTENTION_ALTERNATE, ModelType.RELATION_GROUP_ATTENTION_SELF]
lr_list = [1e-3, 1e-4, 1e-5, 1e-6]
weight_decay_list = [1e-3, 1e-5]
batch_size = 100
epoch_count = 50

###########################################################################################################################
config.BATCH_SIZE = batch_size
config.EPOCH_COUNT = epoch_count
for network_item in network_list:
    config.MODEL_TYPE = network_item
    for dataset_item in dataset_list:
        config.DATALOADER_TYPE = dataset_item
        for lr_item in lr_list:
            for weight_decay_item in weight_decay_list:
                config.LEARNING_RATE = lr_item
                config.WEIGHT_DECAY = weight_decay_item
                train.main()
