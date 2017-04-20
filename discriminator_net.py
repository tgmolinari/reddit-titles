# Discriminator network
# Attempts to learn if the title provided matches the content of the image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo
import torchvision.models as models
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

class ImageDiscriminator(torch.nn.Module):
    def __init__(self, dtype, num_chars):
        super(ImageDiscriminator, self).__init__()

        self.dtype = dtype

        self.title_feat_extractor = nn.GRU(num_chars, 512,)
        self.attn_lin1 = nn.Linear(4096, 512)
        self.attn_conv1 = nn.Conv1d(2,1,1)
        self.lin1 = nn.Linear(512, 256)
        self.lin2 = nn.Linear(256, 1)

        self.type(dtype)

    def forward(self, images, titles, title_lens):

        # because torch doesn't have an argsort, we use numpy's
        sorted_idx = np.array(np.argsort(title_lens.numpy())[::-1])
        sorted_title_lens = title_lens.numpy()[sorted_idx]
        images = images[torch.from_numpy(sorted_idx).cuda()] 
        titles = titles[torch.from_numpy(sorted_idx).cuda()]
        packed_seq = pack_padded_sequence(titles, sorted_title_lens, batch_first = True)

        self.init_hidden(images.size(0))
       
        title_feat, _ = self.title_feat_extractor(packed_seq, self.h0)
        title_feat, lens = pad_packed_sequence(title_feat)
        
        # extracting only last output of GRU
        trimmed_feat = Variable(torch.FloatTensor(title_feat.size(1), title_feat.size(2))).type(self.dtype)
        for batch in range(title_feat.size(1)):
            trimmed_feat[batch] = title_feat[lens[batch] - 1][batch]
        images = self.attn_lin1(images)
        features = torch.cat((trimmed_feat.unsqueeze(1), images.unsqueeze(1)), 1)
        x = self.attn_conv1(features)
        x = self.lin1(x.squeeze(1))
        x = F.leaky_relu(x)
        x = self.lin2(x)
        x = F.sigmoid(x)
        return x[torch.from_numpy(np.argsort(sorted_idx)).cuda()] 

    def init_hidden(self, bsz):
        self.h0 = Variable(torch.zeros(1, bsz, 512).type(self.dtype))

