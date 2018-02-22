# VQA
Deep learning models for visual reasoning on CLEVR, FigureQA and SHAPES datatset

To train a model, first set the parameters in config file correctly and then execute the train.py using the command:
python train.py


clevr -> images -> train, test, val
                   (CLEVR_train_#ID#.png), (CLEVR_test_#ID#.png),
                   (CLEVR_val_#ID#.png)

clevr-> questions ->clevr_train.json
                    clevr_test.json
                    clevr_val.json


shapes->images-> train, test, val
                 (shapes_#id#.png), (shapes_#id#.png), (shapes_#id#.png)
shapes->questions-> shapes_train.json,
                    shapes_test.json,shapes_val.json


FigureQA->images->train, test, val
                    (#id#.png), (#id#.png), (#id#.png)
FigureQA->questions-> FigureQA_test2.json 
                    FigureQA_test.json
                    FigureQA_train.json
                    FigureQA_val2.json
                    FigureQA_val.json


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



SHAPES Stats

Question vocab size = 14
Answer vocab size = 2
question vocab =[u'a', u'blue', u'square', u'triangle', u'of', u'is', u'below', u'shape', u'right', u'green', u'above', u'circle', u'red', u'left']
answer vocab = [u'true', u'false']
Max question length (not including STOP) =11


CLEVR Stats
Question vocab size = 80
Answer vocab size = 28
question vocab =[u'and', u'cylinder', u'right', u'another', u'less', u'color', u'material', u'is', u'spheres', u'it', u'yellow', u'an', u'sphere', u'as', u'ball', u'same', u'have', u'in', u'any', u'size', u'blue', u'what', u'both', u'to', u'purple', u'things', u'there', u'equal', u'tiny', u'metallic', u'rubber', u'how', u'behind', u'other', u'has', u'red', u'more', u'brown', u'cube', u'blocks', u'greater', u'do', u'that', u'big', u'matte', u'object', u'number', u'its', u'made', u'front', u'cyan', u'on', u'than', u'a', u'gray', u'either', u'cubes', u'cylinders', u'anything', u'visible', u'of', u'balls', u'metal', u'shape', u'fewer', u'or', u'large', u'thing', u'else', u'objects', u'green', u'does', u'many', u'small', u'the', u'left', u'shiny', u'side', u'block', u'are']
answer vocab = [u'cylinder', u'yellow', u'sphere', u'yes', u'blue', u'rubber', u'no', u'purple', u'1', u'0', u'3', u'2', u'5', u'4', u'7', u'6', u'9', u'8', u'red', u'brown', u'cube', u'10', u'cyan', u'gray', u'metal', u'large', u'green', u'small']
Max question length (not including STOP) =43


FigureQA Stats
Question vocab size = 84
Answer vocab size = 2
question vocab =[u'cadet', u'indigo', u'have', u'gold', u'less', u'aqua', u'is', u'midnight', u'yellow', u'roughest', u'high', u'pink', u'minimum', u'tomato', u'khaki', u'violet', u'navy', u'tan', u'sky', u'pale', u'magenta', u'sandy', u'orchid', u'blue', u'web', u'peru', u'smoothest', u'sienna', u'area', u'purple', u'mint', u'royal', u'rosy', u'bubblegum', u'chartreuse', u'indian', u'periwinkle', u'low', u'cornflower', u'under', u'orange', u'black', u'lime', u'firebrick', u'red', u'forest', u'lowest', u'brown', u'turquoise', u'medium', u'greater', u'dim', u'slate', u'salmon', u'seafoam', u'chocolate', u'dark', u'crimson', u'deep', u'drab', u'olive', u'cyan', u'highest', u'rebecca', u'than', u'maroon', u'curve', u'steel', u'gray', u'lawn', u'saddle', u'light', u'coral', u'median', u'maximum', u'value', u'hot', u'burlywood', u'green', u'does', u'teal', u'intersect', u'the', u'dodger']
answer vocab = [0, 1]
Max question length (not including STOP) =11

