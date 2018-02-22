'''
This file contains the factory definitons for fetching the appropriate network for training
'''
from enums import ModelType
from baseline import BaselineModel
from film import FiLM
from data_loaders import data_loader_factory
import config

def get_network(network_type):
    '''
    returns an instance of a model with randomly initialized weights
    '''
    dataset_dict = data_loader_factory.get_dataset_dictionary(config.DATALOADER_TYPE)
    if network_type == ModelType.BASELINE:
        return BaselineModel(dataset_dict)
    elif network_type == ModelType.FILM:
        return FiLM(dataset_dict)
    elif network_type == ModelType.STACKED_CO_ATTENTION:
        pass
    elif network_type == ModelType.MEMORY_NETWORK:
        pass
