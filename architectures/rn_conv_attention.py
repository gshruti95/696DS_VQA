import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import config
from enums import DataLoaderType


class ConvInputModel(nn.Module):
    def __init__(self, filter_size):
        super(ConvInputModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, filter_size, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(filter_size)
        self.conv2 = nn.Conv2d(filter_size, filter_size, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(filter_size)
        self.conv3 = nn.Conv2d(filter_size, filter_size, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(filter_size)

        
    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        return x

  
class FCOutputModel(nn.Module):
    def __init__(self, class_count):
        super(FCOutputModel, self).__init__()

        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, class_count)

    def forward(self, x):
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x)

  

class BasicModel(nn.Module):
    def __init__(self, args, name):
        super(BasicModel, self).__init__()
        self.name=name

    def train_(self, input_img, input_qst, label):
        self.optimizer.zero_grad()
        output = self(input_img, input_qst)
        loss = F.nll_loss(output, label)
        loss.backward()
        self.optimizer.step()
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy
        
    def test_(self, input_img, input_qst, label):
        output = self(input_img, input_qst)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy

    def save_model(self, epoch):
        torch.save(self.state_dict(), 'model/epoch_{}_{:02d}.pth'.format(self.name, epoch))


class RelNetConvAttention(nn.Module):
    def __init__(self, dataset_dictionary, architecture_dictionary):
        super(RelNetConvAttention, self).__init__()

        if config.DATALOADER_TYPE != DataLoaderType.SORT_OF_CLEVR:
            self.question_vocab_size = dataset_dictionary[config.QUESTION_VOCAB_SIZE]
            self.answer_vocab_size = dataset_dictionary[config.ANSWER_VOCAB_SIZE]
            self.question_embedding_size = architecture_dictionary[config.QUESTION_EMBEDDING_SIZE]
            self.qlstm_hidden_dim = architecture_dictionary[config.LSTM_HIDDEN_DIM]
            qlstm_num_layers = architecture_dictionary[config.LSTM_LAYERS]

            # construct question embedding
            self.qembedding = nn.Embedding(self.question_vocab_size,
                                       self.question_embedding_size)
            self.qlstm = nn.LSTM(self.question_embedding_size,
                             self.qlstm_hidden_dim,
                             qlstm_num_layers, dropout=0)
        
            ques_dim = self.qlstm_hidden_dim
        else:
            ques_dim = dataset_dictionary[config.QUESTION_EMBEDDING_SIZE]

        self.filter_size = architecture_dictionary[config.FILTER_SIZE]
        self.question_attention_layer_weight = nn.Linear(ques_dim, self.filter_size * self.filter_size * 3 * 3)
        #self.question_attention_layer_bias = nn.Linear(ques_dim, self.filter_size)
        self.question_comparision_layer = nn.Linear(ques_dim, ques_dim)
        
        self.batchNorm4 = nn.BatchNorm2d(self.filter_size)
        
        self.conv = ConvInputModel(self.filter_size)
        
        ##(number of filters per object+coordinate of object)*2+question vector
        self.g_fc1 = nn.Linear((self.filter_size + 2)*2+ ques_dim, 256)

        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)

        self.f_fc1 = nn.Linear(256, 256)

        self.coord_oi = torch.FloatTensor(config.BATCH_SIZE, 2)
        self.coord_oj = torch.FloatTensor(config.BATCH_SIZE, 2)
        if torch.cuda.is_available():
            self.coord_oi = self.coord_oi.cuda()
            self.coord_oj = self.coord_oj.cuda()
        self.coord_oi = Variable(self.coord_oi)
        self.coord_oj = Variable(self.coord_oj)
        
        self.fcout = FCOutputModel(dataset_dictionary[config.ANSWER_VOCAB_SIZE])
        
        self.optimizer = optim.Adam(self.parameters(), lr=config.LEARNING_RATE)

    def update_gradients(self, batch_size, qst):
        grad_list = []
        for conv in self.conv_attention_list:
            grad_list.append(conv.weight.grad.data.unsqueeze(0))
        
        grads = torch.cat(grad_list, dim = 0)
        grads = grads.view(len(grad_list), self.filter_size * self.filter_size * 3 * 3)
        grads = Variable(grads)

        if config.DATALOADER_TYPE != DataLoaderType.SORT_OF_CLEVR:
            ques_embedding = self.qembedding(qst)
            embeddings = ques_embedding.permute(1, 0, 2)
            _, (question_features, _) = self.qlstm(embeddings)
            qst = question_features.view(-1, self.qlstm_hidden_dim)
        else:
            qst = torch.transpose(qst, 0, 1)
            qst = qst.float()

        output = self.question_attention_layer_weight(qst)
        torch.autograd.backward(output, grads)


    def forward(self, img, qst):
        if config.DATALOADER_TYPE != DataLoaderType.SORT_OF_CLEVR:
            ques_embedding = self.qembedding(qst)
            embeddings = ques_embedding.permute(1, 0, 2)
            _, (question_features, _) = self.qlstm(embeddings)
            qst = question_features.view(-1, self.qlstm_hidden_dim)
        else:
            qst = torch.transpose(qst, 0, 1)
            qst = qst.float()
        
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)
        ########################## CONV ATTENTION LOGIC ##############################################################
        n, c, h, w = x.size()
        qst_temp = qst
        qst_attention_weight = self.question_attention_layer_weight(qst_temp)
        qst_attention_weight = F.relu(qst_attention_weight)
        #qst_attention_bias = self.question_attention_layer_bias(qst_temp)
        #qst_attention_bias = F.relu(qst_attention_bias)
        qst_attention_weight = qst_attention_weight.view(n, self.filter_size, self.filter_size, 3, 3)

        # construct multiple convolution layers in a list for each example in the batch
        self.conv_attention_list = [nn.Conv2d(self.filter_size, self.filter_size, 3, stride=2, padding=1, bias=False) for _ in xrange(n)]
        conv_output = []
        for index in xrange(n):
            #print(qst_attention_weight[index].size())
            temp_input = x[index].unsqueeze(0)
            self.conv_attention_list[index].weight.data = qst_attention_weight[index].data
            temp = self.conv_attention_list[index](temp_input)
            conv_output.append(temp)

        x = torch.cat(conv_output, dim =0)
        x = F.relu(x)
        x = self.batchNorm4(x)
        qst_comparison = self.question_comparision_layer(qst)
        qst = F.relu(qst_comparison)
        ##########################CONV ATTENTION LOGIC ###############################################################


        object_count = x.size()[2] * x.size()[3]
        
        """g"""
        mb = x.size()[0]
        n_channels = x.size()[1]
        d = x.size()[2]
        x_flat = x.view(mb,n_channels,d*d).permute(0,2,1)
        def cvt_coord(i):
            return [(i/5-2)/2., (i%5-2)/2.]
        
        # prepare coord tensor
        coord_tensor = torch.FloatTensor(config.BATCH_SIZE, object_count, 2)
        if torch.cuda.is_available():
            coord_tensor = coord_tensor.cuda()
        coord_tensor = Variable(coord_tensor)
        np_coord_tensor = np.zeros((config.BATCH_SIZE, object_count, 2))
        for i in range(object_count):
            np_coord_tensor[:,i,:] = np.array( cvt_coord(i) )
        coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))
        
        # add coordinates
        x_flat = torch.cat([x_flat, coord_tensor],2)
        
        # add question everywhere
        qst = torch.unsqueeze(qst, 1)
        qst = qst.repeat(1,object_count,1)
        qst = torch.unsqueeze(qst, 2)

        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat,1) # (64x1x25x26+11)
        x_i = x_i.repeat(1,object_count,1,1) # (64x25x25x26+11)
        x_j = torch.unsqueeze(x_flat,2) # (64x25x1x26+11)
        x_j = torch.cat([x_j,qst],3)
        x_j = x_j.repeat(1,1,object_count,1) # (64x25x25x26+11)
        # concatenate all together
        x_full = torch.cat([x_i,x_j],3) # (64x25x25x2*26+11)
        # reshape for passing through network
        x_ = x_full.view(mb*d*d*d*d, -1)
        #print(x_.size())
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)
        
        # reshape again and sum
        x_g = x_.view(mb,d*d*d*d,-1)
        #print(x_g.size())
        x_g = x_g.sum(1).squeeze()
        #print(x_g.size())
        
        """f"""
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)
        
        return self.fcout(x_f)
