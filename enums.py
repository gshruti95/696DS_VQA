'''
Thsi file contains all the enumerations used in the code
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

'''
ModelType define the architecture type.
Specified in Config file and used in factory definitions
'''
class ModelType(enumerate):
    BASELINE = 'BASELINE'
    FILM = 'FiLM'
    RELATION_NETWORK = 'RELATION_NETWORK'
    RELATION_NETWORK_CONDITIONAL_BATCH_NORM = 'RELATION NETWORK with conditional batch norm on the input features for last layer'
    STACKED_CO_ATTENTION = 'STACKED_CO_ATTENTION'
    MEMORY_NETWORK = 'MEMORY_NETWORK'