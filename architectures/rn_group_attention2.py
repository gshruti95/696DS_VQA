import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import config

class RelNetGroupAttentionAlternate(nn.Module):
    def __init__(self, dataset_dictionary, architecture_dictionary):
        super(RelNetGroupAttentionAlternate, self).__init__()
        act_f = nn.ReLU()

        self.group_size = architecture_dictionary[config.ATTENTION_GROUP_SIZE]
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
                             qlstm_num_layers,
                             dropout=0)
        ques_dim = self.qlstm_hidden_dim
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
            #nn.Conv2d(img_net_dim, 64, kernel_size=3, stride=2, padding=1),
            #nn.BatchNorm2d(64),
            #act_f,
        )
        img_net_out_dim = img_net_dim # since no pooling operation is performed

        self.attention_layer = nn.Linear(ques_dim + (img_net_out_dim * self.group_size), self.group_size)
        
        g_in_dim = 2 * (img_net_out_dim) + ques_dim # 2 * (img_net_out_dim + 2) + ques_dim
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
        three = ques.unsqueeze(2).repeat(1, 1, n_objects)

        ########################################## Group Attention logic starts ################################################################# 
        # permute the cells in the order (N, objects count, embedding)
        # permute the question in the order (N, objects count, embedding)
        modified_cells = cells.permute(0, 2, 1)
        #print(modified_cells.size())
        modified_n_objects = n_objects / self.group_size # select group size to divide the object count perfectly
        compressed_cells = Variable(torch.zeros(modified_cells.size()[0], modified_n_objects, modified_cells.size()[2]))
        if modified_cells.is_cuda:
            compressed_cells = compressed_cells.cuda()
        j = 0
        while j < modified_n_objects:
            group_features = modified_cells[:, j * self.group_size : (j + 1) * self.group_size, :] # obtain the group size number of features
            # concat all the group features along with the question embedding
            concat_group_features = group_features.contiguous().view(N, -1) # reshape the group features
            concat_group_features = torch.cat([concat_group_features, ques], dim = 1)
            # run through the attention_layer to obtain the temp scores
            temp_scores = self.attention_layer(concat_group_features) #(N, group_size)
            # apply softmax on the temp_scores to perform normalizarion
            normalized_temp_scores = F.softmax(temp_scores)
            normalized_temp_scores = normalized_temp_scores.unsqueeze(2).repeat(1, 1, modified_cells.size()[2])
            # apply the scalar weight on each of the features
            compressed_cells[:, j, :] = torch.sum(normalized_temp_scores * group_features, dim = 1)
            j = j + 1
        # change the order of the objects
        cells = compressed_cells.permute(0, 2, 1)
        # change the number of final objects for comparision
        n_objects = modified_n_objects
        ######################################### Group Attention logic ends ######################################################################
        # append location features to each object/cell
        #loc_feat = self._loc_feat(img)
        #cells = torch.cat([cells, loc_feat], dim=1)
        # accumulate pairwise object embeddings
        pairs = []
        three = ques.unsqueeze(2).repeat(1, 1, n_objects)
        for i in range(n_objects):
            one = cells[:, :, i].unsqueeze(2).repeat(1, 1, n_objects)
            # the number of objects will be reduced here using the attention approach
            # Approach use a single linear layer to to perform weighted sum
            # Approach use multiple linear layers to perform weighted sum 
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
        img = self.img_net(img)
        #print(img.size())
        ques_embedding = self.qembedding(ques)
        embeddings = ques_embedding.permute(1, 0, 2)
        _, (question_features, _) = self.qlstm(embeddings)
        ques = question_features.view(-1, self.qlstm_hidden_dim)
        # RN implementation treating pixels as objects
        # (f and g as in the RN paper)
        context = 0
        pairs = self.img_to_pairs(img, ques)
        N, N_pairs, _ = pairs.size()
        #print(pairs.size())
        context = self.g(pairs.view(N*N_pairs, -1))
        context = context.view(N, N_pairs, -1).mean(dim=1)
        scores = self.f(context)
        return scores #F.log_softmax(scores, dim=1)
