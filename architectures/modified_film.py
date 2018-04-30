import torch
from torch import nn
from torch.nn.init import kaiming_uniform, normal
import torch.nn.functional as F
import config

class CNN(nn.Module):
    def __init__(self, n_blocks, filter_size):
        nn.Module.__init__(self)

        self.conv = nn.ModuleList()

        for i in range(n_blocks):
            if i == 0:
                n_filter = 3

            else:
                n_filter = filter_size

            self.conv.append(nn.Conv2d(n_filter, filter_size,
                                    [4, 4], 2, 1, bias=False))
            self.conv.append(nn.BatchNorm2d(filter_size))
            self.conv.append(nn.ReLU())

        self.reset()

    def reset(self):
        for i in self.conv:
            if isinstance(i, nn.Conv2d):
                kaiming_uniform(i.weight)

    def forward(self, input):
        out = input

        for i in self.conv:
            out = i(out)

        return out

class ResBlock(nn.Module):
    def __init__(self, filter_size):
        nn.Module.__init__(self)

        self.conv1 = nn.Conv2d(filter_size, filter_size, [1, 1], 1, 1)
        self.conv2 = nn.Conv2d(filter_size, filter_size, [3, 3], 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(filter_size, affine=False)

        self.reset()

    def forward(self, input, gamma, beta):
        out = self.conv1(input)
        resid = F.relu(out)
        out = self.conv2(resid)
        out = self.bn(out)
        
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)

        out = gamma * out + beta

        out = F.relu(out)
        out = out + resid
        return out

    def reset(self):
        kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.zero_()
        kaiming_uniform(self.conv2.weight)

class Classifier(nn.Module):
    def __init__(self, n_input=128, n_filter=512, n_hidden=1024, n_layer=2, n_class=28):
        nn.Module.__init__(self)

        self.conv = nn.Conv2d(n_input, n_filter, [1, 1], 1, 1)
        self.mlp = nn.Sequential(nn.Linear(n_filter, n_hidden),
                                nn.ReLU(),
                                nn.Linear(n_hidden, n_hidden),
                                nn.ReLU(),
                                nn.Linear(n_hidden, n_class))

        self.reset()

    def reset(self):
        kaiming_uniform(self.conv.weight)
        self.conv.bias.data.zero_()

        for i in self.mlp:
            if isinstance(i, nn.Linear):
                kaiming_uniform(i.weight)
                i.bias.data.zero_()

    def forward(self, input):
        out = self.conv(input)
        b_size, n_filter, _, _ = out.size()
        out = out.view(b_size, n_filter, -1)
        out, _ = out.max(2)
        out = self.mlp(out)

        return out

class mod_FiLM(nn.Module):
    def __init__(self, dataset_dictionary, n_cnn=4, n_resblock=4, conv_hidden=128, embed_hidden=200, gru_hidden=4096):
       	super(mod_FiLM, self).__init__()
        self.num_question_word = dataset_dictionary[config.QUESTION_VOCAB_SIZE]
        n_vocab = self.num_question_word
        self.conv = CNN(n_cnn, conv_hidden)
        self.resblocks = nn.ModuleList()
        self.res_lins = nn.ModuleList()
        M = [4608, 8192, 12800, 18432]
        for i in range(n_resblock):
            self.resblocks.append(ResBlock(conv_hidden))
            self.res_lins.append(nn.Linear(M[i], 2048))

        self.embed = nn.Embedding(n_vocab, embed_hidden)
        self.gru = nn.GRU(embed_hidden, gru_hidden, batch_first=True)
        self.film = nn.Linear(gru_hidden, conv_hidden * 2)

        self.classifier = Classifier(n_input=conv_hidden, n_class=dataset_dictionary[config.ANSWER_VOCAB_SIZE])

        self.n_resblock = n_resblock
        self.conv_hidden = conv_hidden

        self.rnn = nn.RNNCell(conv_hidden*n_cnn*n_cnn, conv_hidden*2) # input is size of self.conv output , hidden state is size of h output of GRU i.e. 1,N,H --> output is N, conv_hidden * 2 (same as output of self.film)

    def reset(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        kaiming_uniform(self.film.weight)
        self.film.bias.data.zero_()

    def forward(self, image, question):
	question_len = [len(q) for q in question]
        out = self.conv(image) # out is N,C,4,4

        embed = self.embed(question)
        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len,
                                                batch_first=True)
        _, h = self.gru(embed) # h is L,N,H

        lin_h = self.film(h.squeeze()) # lin_h is N,C*2
        rnn_inp = out.view(out.size(0), -1) # N,D

        for i, (resblock, res_lin) in enumerate(zip(self.resblocks, self.res_lins)):
            lin_h = self.rnn(rnn_inp, lin_h) # lin_h is L,N,C*2
            params = lin_h.chunk(2, 1) # list of 2 tensors: gamma and beta each of size N,C
            out = resblock(out, params[0], params[1])
            res_out = out.view(out.size(0), -1)
            #lin = nn.Linear(res_out.size(1), rnn_inp.size(1))
            #lin.cuda()
            #rnn_inp = lin(res_out)
            rnn_inp = res_lin(res_out)

        out = self.classifier(out)
        out = F.log_softmax(out, dim=1)

        return out
