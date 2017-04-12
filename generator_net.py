# Generator network for image -> title
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo
import torchvision.models as models
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

NULL_TERM_IND = 52
MAX_LENGTH = 400

class TitleGenerator(torch.nn.Module):
    def __init__(self, dtype, num_chars):
        super(TitleGenerator, self).__init__()

        self.lin1 = nn.Linear(4096, 128)
        self.char_gen = nn.GRUCell(128, 128)
        self.lin2 = nn.Linear(128, num_chars)
        self.num_chars = num_chars
        self.dtype = dtype
        self.type(dtype)

    def forward(self, img_feat):

        ht = F.leaky_relu(self.lin1(img_feat))
        h0 = self.init_hidden(img_feat.size(0))

        titles = Variable(torch.zeros(MAX_LENGTH, img_feat.size(0), self.num_chars).type(self.dtype))
        for t in range(MAX_LENGTH):
            ht = self.char_gen(ht, h0)
            h0 = ht
            chars = self.lin2(F.leaky_relu(ht))
            for batch in range(chars.size(0)):
                # transforms chars to one hot encoding according to max activation
                chars[batch] = F.threshold(chars[batch] * 1 / torch.max(chars[batch].data), 1-1e-5, 0)
            titles[t] = chars

        titles = torch.transpose(titles, 0, 1)
        title_lens = []
        for batch in range(img_feat.size(0)):
            for t in range(MAX_LENGTH):
                if titles[batch][t][NULL_TERM_IND].data[0] == 1:
                    title_lens.append(t + 1)
                    break
                elif t == MAX_LENGTH - 1:
                    title_lens.append(MAX_LENGTH)

        return titles, torch.LongTensor(title_lens)

    def init_hidden(self, bsz):
        return Variable(torch.zeros(bsz, 128).type(self.dtype))