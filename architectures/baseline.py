'''
This baseline model will consitute a basic CNN to obtain image features and a RNN to obtain the language features 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BaselineMode(nn.Module):

    def __init__(self, **params):
        pass


    def forward(self, input):
        pass