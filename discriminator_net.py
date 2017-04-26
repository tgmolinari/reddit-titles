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
    def __init__(self, dtype, num_dims):
        super(ImageDiscriminator, self).__init__()

        self.dtype = dtype

        self.title_feat_extractor = nn.GRU(num_dims, 512, batch_first = True)
        self.attn_lin1 = nn.Linear(4096, 512)
        self.attn_conv1 = nn.Conv1d(2,1,1)
        self.lin1 = nn.Linear(512, 256)
        self.lin2 = nn.Linear(256, 1)

        self.type(dtype)

    def forward(self, images, titles):

        self.init_hidden(images.size(0))
       
        title_feat, _ = self.title_feat_extractor(titles, self.h0)
        
        # extracting only last output of GRU
        trimmed_feat = title_feat[:,-1]
            
        images = self.attn_lin1(images)
        features = torch.cat((trimmed_feat.unsqueeze(1), images.unsqueeze(1)), 1)
        x = self.attn_conv1(features)
        x = self.lin1(x.squeeze(1))
        x = F.leaky_relu(x)
        x = self.lin2(x)
        x = F.sigmoid(x)
        return x 

    def init_hidden(self, bsz):
        self.h0 = Variable(torch.zeros(1, bsz, 512).type(self.dtype))

