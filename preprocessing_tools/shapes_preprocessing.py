'''
This script will read the shapes_dataset files from (/SHAPES/n2nmn/exp_shapes/shapes_dataset) and contruct of a hierarchy of folder for the images and 
question like other datasets
'''

import sys
import os
import json
import numpy as np
from scipy import misc


def get_mode(MODE_NAME):
    mode = ''
    if MODE_NAME.find('train') > -1:
        mode = 'train'
    elif MODE_NAME.find('val') > -1:
        mode = 'val'
    else:
        mode = 'test'
    return mode


def generate_images_folder(FOLDER_PATH, MODE_NAME):
    IMAGE_FOLDER_PATH = FOLDER_PATH + 'images'
    IMAGE_FILE_PREFIX = 'shapes_'
    IMAGE_FILE_FORMAT = '.png'
    if os.path.exists(IMAGE_FOLDER_PATH) == False:
        os.mkdir(IMAGE_FOLDER_PATH)
    
    mode = get_mode(MODE_NAME)
    FINAL_FOLDER_PATH = IMAGE_FOLDER_PATH + '/' + mode
    if os.path.exists(FINAL_FOLDER_PATH) == False:
        os.mkdir(FINAL_FOLDER_PATH)

    numpy_object_name = FOLDER_PATH + MODE_NAME + 'input.npy'
    numpy_object = np.load(numpy_object_name)
    print(numpy_object.shape)
    for image_index in xrange(numpy_object.shape[0]):
        image = numpy_object[image_index, :, :, :]
        image_name = IMAGE_FILE_PREFIX + str(image_index) + IMAGE_FILE_FORMAT
        misc.imsave(FINAL_FOLDER_PATH + '/' + image_name, image)


def generate_questions_folder(FOLDER_PATH, MODE_NAME):
    IMAGE_FILE_PREFIX = 'shapes_'
    IMAGE_FILE_FORMAT = '.png'
    QUESTION_FILE_PREFIX = '.json'
    QUESTION_FOLDER_PATH = FOLDER_PATH + 'questions'
    if os.path.exists(QUESTION_FOLDER_PATH) == False:
        os.mkdir(QUESTION_FOLDER_PATH)

    mode = get_mode(MODE_NAME)
    question_output_filepath = FOLDER_PATH + MODE_NAME + 'output'
    question_str_filepath = FOLDER_PATH + MODE_NAME + 'query_str.txt'
    question_output = open(question_output_filepath, 'r').read()
    question_output_list = question_output.split('\n')
    question_str = open(question_str_filepath, 'r').read()
    question_str_list = question_str.split('\n')

    modified_list = []

    for index in xrange(len(question_output_list)):
        query_item = question_str_list[index]
        answer_item = question_output_list[index]
        image_file_name = IMAGE_FILE_PREFIX + str(index) + IMAGE_FILE_FORMAT
        modfiied_dict = {'question' : query_item, 'answer' : answer_item, 'image_filename' : image_file_name}
        modified_list.append(modfiied_dict)

    json_data = json.dumps(modified_list)
    question_json_file = open(QUESTION_FOLDER_PATH + '/' + IMAGE_FILE_PREFIX + mode + QUESTION_FILE_PREFIX, 'w')
    question_json_file.writelines(json_data)
    question_json_file.close()


def main():
    FOLDER_PATH = '../../datasets/SHAPES/n2nmn/exp_shapes/shapes_dataset/'
    
    MODE_NAME = 'train.large.'
    generate_images_folder(FOLDER_PATH, MODE_NAME)
    print('Generated train images folder')
    generate_questions_folder(FOLDER_PATH, MODE_NAME)
    print('Generated train questions json')

    MODE_NAME = 'val.'
    generate_images_folder(FOLDER_PATH, MODE_NAME)
    print('Generated val images folder')
    generate_questions_folder(FOLDER_PATH, MODE_NAME)
    print('Generated val questions json')

    MODE_NAME = 'test.'
    generate_images_folder(FOLDER_PATH, MODE_NAME)
    print('Generated test images folder')
    generate_questions_folder(FOLDER_PATH, MODE_NAME)
    print('Generated test questions json')


if __name__ == '__main__':
    main()