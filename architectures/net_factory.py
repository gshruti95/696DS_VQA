'''
This file contains the factory definitons for fetching the appropriate network for training
'''
from enums import ModelType
from baseline import BaselineModel
from rn import RelNet
from film import FiLM
from modified_film import mod_FiLM
from bn_modified_film import bn_mod_FiLM
from data_loaders import data_loader_factory
import config
from rn_group_attention_standard import RelNetGroupAttentionStandard
from rn_group_attention_alternate import RelNetGroupAttentionAlternate
from rn_group_attention_self import RelNetGroupAttentionSelf
from rn_batch_norm import RelNetBatchNorm
from rn_conv_attention import RelNetConvAttention

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
        return FiLM(dataset_dict)
    elif network_type == ModelType.mod_FILM:
        return mod_FiLM(dataset_dict)
    elif network_type == ModelType.bn_mod_FILM:
        return bn_mod_FiLM(dataset_dict)
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
    elif network_type == ModelType.RELATION_NETWORK_BATCH_NORM:
        return RelNetBatchNorm(dataset_dict, config.RELATION_NETWORK_DICTIONARY)
    elif network_type == ModelType.RELATION_NETWORK_CONV_ATTENTION:
        return RelNetConvAttention(dataset_dict, config.RELATION_NETWORK_DICTIONARY)
