'''
This file provides the class definition for the CLEVR Dataset
'''
import sys
import os
import torch
import torch.utils.data as data
from scipy import misc
import config
from skimage.transform import resize
import utilities


class CLEVRDataset(data.Dataset):
    '''
    Provides deifnitions for the CLEVR dataset loader.
    '''
    def __init__(self, data_mode, data_filepath):
        # load all the filenames from the dataset folder based on the data mode(train, val, test)
        self.target_filepath = data_filepath + data_mode + '_text/'
        self.file_names = os.listdir(self.target_filepath)

    
    def __getitem__(self, index):
        # returns (X, y) tuple
        current_sample_filepath = self.file_names[index]
        image = misc.imread(self.target_filepath + current_sample_filepath)
        # need to transform the image to the input size of the net
        net_input_xsize = config.MODEL_INPUT_SIZE[0]
        net_input_ysize = config.MODEL_INPUT_SIZE[1]  
        x = resize(image, (net_input_xsize, net_input_ysize), mode = 'constant')
        # extract the y label from the file_name
        file_items = current_sample_filepath.replace(config.IMAGE_FILETYPE, '').split('_')
        y_string = file_items[-1]
        # construct the class wise representation of each character before returning
        label = utilities.encode_string(y_string)
        return (x, label)

    def __len__(self):
        return len(self.file_names)