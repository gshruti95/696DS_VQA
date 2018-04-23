'''
This script constructs new json file from the existing qa_pairs json file to construct the standard json file for model training.
Will not perform any image preprocessing. Manually place the images in the correct folders for training
'''

import sys
import os
import json


def generate_json_file(QUESTION_FOLDER_PATH, TARGET_FILE_NAME, json_file_list):
    modified_list = []
    IMAGE_FILE_TYPE = '.png'
    for json_file in json_file_list:
        json_data = file(QUESTION_FOLDER_PATH + json_file).read()
        data = json.loads(json_data)
        qa_pairs_list = data['qa_pairs']
        for qa_item in qa_pairs_list:
            modified_dict = {}
            modified_dict['question'] = qa_item['question_string']
            modified_dict['answer'] = qa_item['answer'] #Returns 0 or 1
            modified_dict['image_filename'] = str(qa_item['image_index']) + IMAGE_FILE_TYPE
            modified_dict['question_type'] = qa_item['question_id']
            modified_list.append(modified_dict) 
    question_json_file = open(QUESTION_FOLDER_PATH + TARGET_FILE_NAME, 'w')
    json_data = json.dumps(modified_list)
    question_json_file.writelines(json_data)
    question_json_file.close()
        
        
def main():
    QUESTION_FOLDER_PATH = '../datasets/FigureQA/validation2/'
    TARGET_FILE_NAME = 'FigureQA_val2.json'
    json_file_list = ['qa_pairs.json']
    generate_json_file(QUESTION_FOLDER_PATH, TARGET_FILE_NAME, json_file_list)


if __name__ == '__main__':
    main()