'''
This file contains code to extract the language metrics from the question json files
'''
import os
import sys
import json


def main():
    json_filepath = sys.argv[1]
    json_data = file(json_filepath).read()
    question_list = json.loads(json_data)
    word_dictionary = {}
    answer_dictionary = {}
    max_word_length = 0
    for item in question_list:
        #print item
        question = item['question']
        answer = item['answer']
        word_list = question.replace('?','').replace(';','').lower().split(' ')
        if max_word_length < len(word_list):
            max_word_length = len(word_list)
        for word in word_list:
            if word == '':
                continue
            word_dictionary[word] = True
        if answer == '':
            continue
        answer_dictionary[answer] = True
    
    print('Question vocab size = ' + str(len(word_dictionary.keys())))
    print('Answer vocab size = ' + str(len(answer_dictionary.keys())))
    print('question vocab =' + str(word_dictionary.keys()))
    print('answer vocab = ' +  str(answer_dictionary.keys()))
    print('Max question length (not including STOP) =' + str(max_word_length))



if __name__ == '__main__':
    main()
 