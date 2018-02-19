'''
This script reads the question json file and returns a modified json file that will be maintainted as a standard across all datatsets.

The final JSON file has the following format:
[
    {
    'question': Are there any other things that are the same shape as the big metallic object?,
    'answer': 'no',
    'image_filename': 'CLEVR_val_000000.png'
    }

    {
    'question': 'Are there any other things that are the same shape as the big green thing?',
    'answer': 'no',
    'image_filename': 'CLEVR_val_000000.png'
    }
]


The input JSON file is a question JSON file.
Each of the question files has the following format:

{
  "info": <info>,
  "questions": [<question>]
}

<info> {
  "split": <string: "train", "val", or "test">,
  "version": <string>,
  "date": <string>,
  "license": <string>
}

<question> {
  "split": <string: "train", "val", or "test">,
  "image_index": <integer>,
  "image_filename": <string, e.g. "CLEVR_train_000000.png">,
  "question": <string>,
  "answer": <string>,
  "program": [<function>],
  "question_family_index': <integer>,
}

The CLEVR dataset uses the following output labels as answers:
[u'cylinder', u'yellow', u'sphere', u'yes', u'blue', u'rubber', u'no', u'purple',
u'1', u'0', u'3', u'2', u'5', u'4', u'7', u'6', u'9', u'8', u'red', u'brown', u'cube',
u'10', u'cyan', u'gray', u'metal', u'large', u'green', u'small']
'''


import sys
import os
import json

def process_input_json(input_json_filepath):
    json_data = file(input_json_filepath).read()
    data = json.loads(json_data)
    questions = data['questions']
    modified_list = []
    answer_dict ={}
    for item in questions:
        modified_dict = {}
        modified_dict['question'] = item['question']
        if item.has_key('answer') == True:
            modified_dict['answer'] = item['answer']
            answer_dict[item['answer']] = True
        modified_dict['image_filename'] = item['image_filename']
        modified_list.append(modified_dict)
        
    print(answer_dict.keys())
    json_data = json.dumps(modified_list)
    return json_data


def write_output_json_file(json_data, output_json_filepath):
    file = open(output_json_filepath, 'w')
    file.writelines(json_data)
    file.close()


def main():
    input_json_filepath = sys.argv[1]
    output_json_filepath = sys.argv[2]
    if len(sys.argv) != 3:
        print('Passed invalid number of arguments. Enter input_json_filepath and output_json_filepath')
    
    output_json_data = process_input_json(input_json_filepath)
    write_output_json_file(output_json_data, output_json_filepath)


if __name__ == '__main__':
    main()