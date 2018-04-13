'''
This file contains the factory definitons for fetching the appropriate network for training
'''
from enums import ModelType
from baseline import BaselineModel
from rn import RelNet
from data_loaders import data_loader_factory
import config
from rn_group_attention_standard import RelNetGroupAttentionStandard
from rn_group_attention_alternate import RelNetGroupAttentionAlternate
from rn_group_attention_self import RelNetGroupAttentionSelf

def get_network(network_type):
    '''
    returns an instance of a model with randomly initialized weights
    '''
    dataset_dict = data_loader_factory.get_dataset_dictionary(config.DATALOADER_TYPE)
    if network_type == ModelType.BASELINE:
        return BaselineModel(dataset_dict)
    elif network_type == ModelType.RELATION_NETWORK:
        return RelNet(dataset_dict, config.RELATION_NETWORK_DICTIONARY)
    elif network_type == ModelType.FILM:
        pass
    elif network_type == ModelType.STACKED_CO_ATTENTION:
        pass
    elif network_type == ModelType.MEMORY_NETWORK:
        pass
    elif network_type == ModelType.RELATION_GROUP_ATTENTION_STANDARD:
        return RelNetGroupAttentionStandard(dataset_dict, config.RELATION_GROUP_ATTENTION_NETWORK_DICTIONARY)
    elif network_type == ModelType.RELATION_GROUP_ATTENTION_ALTERNATE:
        return RelNetGroupAttentionAlternate(dataset_dict, config.RELATION_GROUP_ATTENTION_NETWORK_DICTIONARY)
    elif network_type == ModelType.RELATION_GROUP_ATTENTION_SELF:
        return RelNetGroupAttentionSelf(dataset_dict, config.RELATION_GROUP_ATTENTION_NETWORK_DICTIONARY)
