'''
This baseline model will consitute a basic CNN to obtain image features and a RNN to obtain the language features 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import config

class BaselineModel(nn.Module):

    def __init__(self, dataset_dictionary):
        super(BaselineModel, self).__init__()
        self.image_size = dataset_dictionary[config.IMAGE_SIZE]
        self.num_question_word = dataset_dictionary[config.QUESTION_VOCAB_SIZE]
        self.num_output_labels = dataset_dictionary[config.ANSWER_VOCAB_SIZE]
        self.lstm_hidden_state_size = 128
        self.word_embedding_size = 128
        self.embedding_matrix = nn.Embedding(self.num_question_word, self.word_embedding_size)
        self.question_lstm = nn.LSTM(self.word_embedding_size, self.lstm_hidden_state_size, 1)
        
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #nn.Dropout2d(p = 0.3)
        )

        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #nn.Dropout(p = 0.3)
        )

        self.linear_layer1 = nn.Linear(64 * 2 * 2, 128)
        self.linear_layer2 = nn.Linear(256, self.num_output_labels)



    def forward(self, images, questions):
        features = self.conv_layer1(images)
        features = self.conv_layer2(features)
        features = self.conv_layer3(features)
        features = features.view(-1, 256)
        image_features = F.relu(self.linear_layer1(features))

        embeddings = self.embedding_matrix(questions)
        embeddings = embeddings.permute(1, 0, 2)
        h0 = Variable(torch.randn((1, self.lstm_hidden_state_size)))
        c0 = Variable(torch.randn((1, self.lstm_hidden_state_size)))
        _, (question_features, _) = self.question_lstm(embeddings, (h0, c0))
        question_features = question_features.view(-1, self.lstm_hidden_state_size)
        combined_features = torch.cat((image_features, question_features), dim = 1)
        output = self.linear_layer2(combined_features)
        return output