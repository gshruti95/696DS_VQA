'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import config
from scipy import misc


class ConvInputModel(nn.Module):
    def __init__(self):
        super(ConvInputModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(24)

        
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
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        return x

  
class FCOutputModel(nn.Module):
    def __init__(self):
        super(FCOutputModel, self).__init__()

        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

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


class RelNet(nn.Module):
    def __init__(self, dataset_dict, architecture_dict):
        super(RelNet, self).__init__()
        
        self.conv = ConvInputModel()
        
        ##(number of filters per object+coordinate of object)*2+question vector
        self.g_fc1 = nn.Linear((24+2)*2+11, 256)

        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)

        self.f_fc1 = nn.Linear(256, 256)

        self.coord_oi = torch.FloatTensor(config.BATCH_SIZE, 2)
        self.coord_oj = torch.FloatTensor(config.BATCH_SIZE, 2)
        #if args.cuda:
        #    self.coord_oi = self.coord_oi.cuda()
        #    self.coord_oj = self.coord_oj.cuda()
        self.coord_oi = Variable(self.coord_oi)
        self.coord_oj = Variable(self.coord_oj)

        # prepare coord tensor
        def cvt_coord(i):
            return [(i/5-2)/2., (i%5-2)/2.]
        
        self.coord_tensor = torch.FloatTensor(config.BATCH_SIZE, 25, 2)
        #if args.cuda:
        #    self.coord_tensor = self.coord_tensor.cuda()
        self.coord_tensor = Variable(self.coord_tensor)
        np_coord_tensor = np.zeros((config.BATCH_SIZE, 25, 2))
        for i in range(25):
            np_coord_tensor[:,i,:] = np.array( cvt_coord(i) )
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))


        self.fcout = FCOutputModel()
        
        self.optimizer = optim.Adam(self.parameters(), lr=config.LEARNING_RATE)


    def forward(self, img, qst):
        qst = torch.transpose(qst, 1, 0)
        #np.set_printoptions(threshold=np.nan)
        #temp = img.data[0].numpy()
        #misc.imshow(temp)
        #print(img.size())
        #print(qst.size())
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)
        
        
        """g"""
        mb = x.size()[0]
        n_channels = x.size()[1]
        d = x.size()[2]
        # x_flat = (64 x 25 x 24)
        x_flat = x.view(mb,n_channels,d*d).permute(0,2,1)
        
        # add coordinates
        x_flat = torch.cat([x_flat, self.coord_tensor],2)
        
        # add question everywhere
        qst = torch.unsqueeze(qst, 1)
        qst = qst.repeat(1,25,1)
        qst = torch.unsqueeze(qst, 2)
        
        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat,1) # (64x1x25x26+11)
        x_i = x_i.repeat(1,25,1,1) # (64x25x25x26+11)
        x_j = torch.unsqueeze(x_flat,2) # (64x25x1x26+11)
        x_j = torch.cat([x_j,qst],3)
        x_j = x_j.repeat(1,1,25,1) # (64x25x25x26+11)
        
        # concatenate all together
        x_full = torch.cat([x_i,x_j],3) # (64x25x25x2*26+11)
        
        # reshape for passing through network
        x_ = x_full.view(mb*d*d*d*d,63)
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)
        
        # reshape again and sum
        x_g = x_.view(mb,d*d*d*d,256)
        x_g = x_g.sum(1).squeeze()
        
        """f"""
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)
        
        return self.fcout(x_f)


class CNN_MLP(BasicModel):
    def __init__(self, args):
        super(CNN_MLP, self).__init__(args, 'CNNMLP')

        self.conv  = ConvInputModel()
        self.fc1   = nn.Linear(5*5*24 + 11, 256)  # question concatenated to all
        self.fcout = FCOutputModel()

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        #print([ a for a in self.parameters() ] )
  
    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)

        """fully connected layers"""
        x = x.view(x.size(0), -1)
        
        x_ = torch.cat((x, qst), 1)  # Concat question
        
        x_ = self.fc1(x_)
        x_ = F.relu(x_)
        
        return self.fcout(x_)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import config
import numpy as np

class RelNet(nn.Module):
    def __init__(self, dataset_dictionary, architecture_dictionary):
        super(RelNet, self).__init__()
        act_f = nn.ReLU()
        self.answer_vocab_size = dataset_dictionary[config.REL_ANSWER_VOCAB_SIZE]
        #self.question_embedding_size = dataset_dictionary[config.QUESTION_EMBEDDING_SIZE]
        # construct question embedding
        #self.qembedding = nn.Embedding(self.question_vocab_size,
        #                               self.question_embedding_size)
        #self.qlstm = nn.LSTM(self.question_embedding_size,
        #                     self.qlstm_hidden_dim,
        #                     qlstm_num_layers,
        #                     dropout=0)
        ques_dim = dataset_dictionary[config.QUESTION_EMBEDDING_SIZE]
        # construct image embeddings
        img_net_dim = dataset_dictionary[config.IMAGE_SIZE]
        self.img_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            act_f,
            nn.Conv2d(64, img_net_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(img_net_dim),
            act_f,
            nn.Conv2d(img_net_dim, img_net_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(img_net_dim),
            act_f,
            nn.Conv2d(img_net_dim, img_net_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(img_net_dim),
            act_f,
            nn.Conv2d(img_net_dim, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            act_f,
        )
        img_net_out_dim = img_net_dim # since no pooling operation is performed
        
        g_in_dim = 2 * (img_net_out_dim + 2) + ques_dim
        #print(g_in_dim)
        rn_f_layer_dim = architecture_dictionary[config.F_LAYER_DIM]
        rn_g_layer_dim = architecture_dictionary[config.G_LAYER_DIM]
        # To add batchnorm or not
        if architecture_dictionary[config.ENABLE_BATCHNORM] == True:
            f_act = nn.Sequential(
                    nn.BatchNorm1d(rn_f_layer_dim),
                    act_f,
                )
            g_act = nn.Sequential(
                    nn.BatchNorm1d(rn_g_layer_dim),
                    act_f,
                )
        else:
                f_act = g_act = act_f
        self.g = nn.Sequential(
            nn.Linear(g_in_dim, rn_g_layer_dim),
            g_act,
            nn.Linear(rn_g_layer_dim, rn_g_layer_dim),
            g_act,
            nn.Linear(rn_g_layer_dim, rn_g_layer_dim),
            g_act,
            nn.Linear(rn_g_layer_dim, rn_g_layer_dim),
            g_act,
        )
        self.f = nn.Sequential(
            nn.Linear(rn_g_layer_dim, rn_f_layer_dim),
            f_act,
            nn.Linear(rn_f_layer_dim, rn_f_layer_dim),
            f_act,
            nn.Dropout(),
            nn.Linear(rn_f_layer_dim, self.answer_vocab_size),
        )
        self.loc_feat_cache = {}
        # random init
        self.apply(self.init_parameters)

    @staticmethod
    def init_parameters(mod):
        if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
            nn.init.kaiming_uniform(mod.weight)
            if mod.bias is not None:
                nn.init.constant(mod.bias, 0)

    def img_to_pairs(self, img, ques):
        '''
        Take a small feature map `img` (say 8x8), treating each pixel
        as an object, and return a tensor with one feature
        per pair of objects.
        Arguments:
            img: tensor of size (N, C, H, W) with CNN features of an image
            ques: tensor of size (N, E) containing question embeddings
        Returns:
            Tensor of size (N, num_pairs=HW*HW, feature_dim=2C + E + 2)
        '''
        N, _, H, W = img.size()
        n_objects = H * W
        cells = img.view(N, -1, n_objects)
        # append location features to each object/cell
        loc_feat = self._loc_feat(img)
        cells = torch.cat([cells, loc_feat], dim=1)
        # accumulate pairwise object embeddings
        pairs = []
        three = ques.unsqueeze(2).repeat(1, 1, n_objects)
        for i in range(n_objects):
            one = cells[:, :, i].unsqueeze(2).repeat(1, 1, n_objects)
            two = cells
            # N x C x n_pairs
            i_pairs = torch.cat([one, two, three], dim=1)
            pairs.append(i_pairs)
        pairs = torch.cat(pairs, dim=2)
        result = pairs.transpose(1, 2).contiguous()
        return result

    def _loc_feat(self, img):
        '''
        Efficiently compute a feature specifying the numeric coordinates of
        each object (pair of pixels) in img.
        '''
        N, _, H, W = img.size()
        key = (N, H, W)
        if key not in self.loc_feat_cache:
            # constant features get appended to RN inputs, compute these here
            loc_feat = torch.FloatTensor(N, 2, W**2)
            if img.is_cuda:
                loc_feat = loc_feat.cuda()
            for i in range(W**2):
                loc_feat[:, 0, i] = i // W
                loc_feat[:, 1, i] = i % W
            self.loc_feat_cache[key] = Variable(loc_feat)
        return self.loc_feat_cache[key]

    def forward(self, img, ques):
        ques = torch.transpose(ques, 0, 1)
        #temp = img.data[0].numpy()
        #np.set_printoptions(threshold = np.nan)
        #print(ques.size())
        #print(temp)
        img = self.img_net(img)
        #print(img.size())
        #ques_embedding = self.qembedding(ques)
        #embeddings = ques_embedding.permute(1, 0, 2)
        #_, (question_features, _) = self.qlstm(embeddings)
        #ques = question_features.view(-1, self.qlstm_hidden_dim)
        # RN implementation treating pixels as objects
        # (f and g as in the RN paper)
        context = 0
        pairs = self.img_to_pairs(img, ques)
        N, N_pairs, _ = pairs.size()
        #print(pairs.size())
        context = self.g(pairs.view(N*N_pairs, -1))
        context = context.view(N, N_pairs, -1).mean(dim=1)
        scores = self.f(context)
        return F.log_softmax(scores)

