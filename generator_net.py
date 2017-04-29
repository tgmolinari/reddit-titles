# Generator network for image -> title
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo
import torchvision.models as models
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

MAX_LENGTH = 100

class TitleGenerator(torch.nn.Module):
    def __init__(self, dtype, num_dims):
        super(TitleGenerator, self).__init__()

        self.lin1 = nn.Linear(4096, 128)
        self.word_gen = nn.GRUCell(138, 138) # 128 plus length 10 noise
        self.lin2 = nn.Linear(138, 128)
        self.lin3 = nn.Linear(128, num_dims)
        self.num_dims = num_dims
        self.dtype = dtype
        self.type(dtype)

    def forward(self, img_feat):

        ht = F.tanh(self.lin1(img_feat))
        noise = Variable(torch.randn(img_feat.size(0), 10)).type(self.dtype)
        ht = torch.cat((ht, noise), 1) # adding some noise
        h0 = self.init_hidden(img_feat.size(0))

        titles = Variable(torch.zeros(MAX_LENGTH, img_feat.size(0), self.num_dims).type(self.dtype))
        for t in range(MAX_LENGTH):
            ht = self.word_gen(ht, h0)
            h0 = ht
            embeddings = self.lin2(F.leaky_relu(ht))
            embeddings = self.lin3(embeddings)
            titles[t] = embeddings


        return titles

    def init_hidden(self, bsz):
        return Variable(torch.zeros(bsz, 138).type(self.dtype))