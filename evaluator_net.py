# Evaluator network for image, title pairs
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo
import torchvision.models as models
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

class ScorePredictor(torch.nn.Module):
    def __init__(self, dtype, num_chars):
        super(ScorePredictor, self).__init__()

        self.dtype = dtype
        self.init_cnn()
        self.title_feat_extractor = nn.GRU(num_chars, 512,)
        self.lin1 = nn.Linear(4096 + 512, 256)
        self.lin2 = nn.Linear(256, 1)

        self.type(dtype)

    def forward(self, images, titles, title_lens):

        # because torch doesn't have an argsort, we use numpy's
        sorted_idx = np.array(np.argsort(title_lens.numpy())[::-1])
        sorted_title_lens = title_lens.numpy()[sorted_idx]
        images = images[torch.from_numpy(sorted_idx).cuda()] # TODO: make this compatible with CPU
        titles = titles[torch.from_numpy(sorted_idx).cuda()]
        packed_seq = pack_padded_sequence(titles, sorted_title_lens, batch_first = True)

        img_feat = self.img_feat_extractor(images)
        self.init_hidden(images.size(0))
       
        title_feat, _ = self.title_feat_extractor(packed_seq, self.h0)
        title_feat, lens = pad_packed_sequence(title_feat)
        
        # extracting only last output of GRU
        trimmed_feat = Variable(torch.FloatTensor(title_feat.size(1), title_feat.size(2))).type(self.dtype)
        for batch in range(title_feat.size(1)):
            trimmed_feat[batch] = title_feat[lens[batch] - 1][batch]
        
        features = torch.cat((trimmed_feat, img_feat), 1)
        x = F.leaky_relu(self.lin1(features))
        x = self.lin2(x)
        return x[torch.from_numpy(np.argsort(sorted_idx)).cuda()] # TODO: this too 

    def init_hidden(self, bsz):
        self.h0 = Variable(torch.zeros(1, bsz, 512).type(self.dtype))

    def init_cnn(self):
        # Modified forward for VGG so that it acts as a feature extractor
        def _forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier[0](x)
            return x

        self.img_feat_extractor = models.vgg16()

        model_url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
        vgg_state_dict = torch.utils.model_zoo.load_url(model_url)
        self.img_feat_extractor.load_state_dict(vgg_state_dict)
        
        for param in self.img_feat_extractor.parameters():
            param.requires_grad = False

        self.img_feat_extractor.type(self.dtype)

        # Monkeywrenching to decapitate the pretrained net
        fType = type(self.img_feat_extractor.forward)
        self.img_feat_extractor.forward = fType(_forward,self.img_feat_extractor)
