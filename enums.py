'''
This file contains all the enumerations used in the code
'''

import sys
import os

class DataMode(enumerate):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'
'''
DataLoader type to be specified in the config file and used in the factory definitions
'''
class DataLoaderType(enumerate):
    CLEVR = 'CLEVR'
    FIGUREQA = 'FigureQA'
    SHAPES = 'SHAPES'
    SORT_OF_CLEVR = 'SORT OF CLEVR'

'''
ModelType define the architecture type.
Specified in Config file and used in factory definitions
'''
class ModelType(enumerate):
    BASELINE = 'BASELINE'
    FILM = 'FiLM'
    MOD_FILM = 'mod_FiLM'
    BN_MOD_FILM = 'bn_mod_FiLM'
    RELATION_NETWORK = 'RELATION_NETWORK'
    STACKED_CO_ATTENTION = 'STACKED_CO_ATTENTION'
    MEMORY_NETWORK = 'MEMORY_NETWORK'
    RELATION_GROUP_ATTENTION_STANDARD = 'RELATION_GROUP_ATTENTION_STANDARD'
    RELATION_GROUP_ATTENTION_ALTERNATE = 'RELATION_GROUP_ATTENTION_ALTERNATE'
    RELATION_GROUP_ATTENTION_SELF = 'RELATION_GROUP_ATTENTION_SELF'
    RELATION_NETWORK_BATCH_NORM = 'RELATION_NETWORK_BATCH_NORM'
    RELATION_NETWORK_CONV_ATTENTION = 'RELATION_NETWORK_CONV_ATTENTION'
