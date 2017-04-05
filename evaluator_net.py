# Evaluator network for image, title pairs

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo
import torchvision.models as models
from torch.nn.utils.rnn import pad_packed_sequence
from torch.autograd import Variable

class ScorePredictor(torch.nn.Module):
    def __init__(self, dtype, num_chars, pretrained=True):
        super(ScorePredictor, self).__init__()

        self.dtype = dtype
        self.init_cnn(pretrained)
        self.title_feat_extractor = nn.GRU(num_chars, 512,)
        self.lin1 = nn.Linear(4096 + 512, 256)
        self.lin2 = nn.Linear(256, 1)

        self.type(dtype)

    def forward(self, imgs, titles):
        img_feat = self.img_feat_extractor(imgs)
        self.init_hidden(imgs.size(0))
       
        title_feat, _ = self.title_feat_extractor(titles, self.h0)
        title_feat, lens = pad_packed_sequence(title_feat)
        trimmed_feat = Variable(torch.FloatTensor(title_feat.size(1), title_feat.size(2))).type(dtype)
        for batch in range(title_feat.size(1)):
            trimmed_feat[batch] = title_feat[lens[batch] - 1][batch]
        
        features = torch.cat((trimmed_feat, img_feat), 1)
        x = F.elu(self.lin1(features))
        return self.lin2(x)

    def init_hidden(self, bsz):
        self.h0 = Variable(torch.zeros(1, bsz, 512).type(self.dtype))

    def init_cnn(self, pretrained):
        # Modified forward for VGG so that it acts as a feature extractor
        def _forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier[0](x)
            return x

        self.img_feat_extractor = models.vgg16()

        if pretrained:
            model_url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
            vgg_state_dict = torch.utils.model_zoo.load_url(model_url)
            self.img_feat_extractor.load_state_dict(vgg_state_dict)

        self.img_feat_extractor.type(self.dtype)

        # Monkeywrenching to decapitate the pretrained net
        fType = type(self.img_feat_extractor.forward)
        self.img_feat_extractor.forward = fType(_forward,self.img_feat_extractor)
