'''
This file contains the factory definitons for fetching the appropriate network for training
'''
from ..enums import ModelType
from baseline import BaselineModel

def get_network(network_type):
    '''
    returns an instance of a model with randomly initialized weights
    '''
    if network_type == ModelType.BASELINE:
        return BaselineModel()
    elif network_type == ModelType.FILM:
        pass
    elif network_type == ModelType.STACKED_CO_ATTENTION:
        pass
    elif network_type == ModelType.MEMORY_NETWORK:
        pass
