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

        self.init_cnn()
        self.lin1 = nn.Linear(4096, num_chars)
        self.char_gen = nn.GRUCell(num_chars, num_chars)
        self.lin2 = nn.Linear(num_chars, num_chars)
        self.num_chars = num_chars
        self.dtype = dtype
        self.type(dtype)

    def forward(self, images):

        img_feat = self.img_feat_extractor(images)
        x = F.leaky_relu(self.lin1(img_feat))

        ht = self.init_hidden(images.size(0))
        chars = x

        titles = torch.zeros(images.size(0), MAX_LENGTH, self.num_chars)
        for t in range(MAX_LENGTH):
            chars, ht = self.char_gen(chars, ht)
            chars = self.lin2(F.leaky_relu(chars))
            for batch in range(chars.size(0)):
                # transforms chars to one hot encoding according to max activation
                chars[batch] = F.threshold(chars[batch] * 1 / torch.max(chars[batch]), 1, 0)
            titles[:][t] = chars

        title_lens = []
        for batch in range(images.size(0)):
            for t in range(MAX_LENGTH):
                _, ind = torch.max(titles[batch][t])
                if ind == NULL_TERM_IND:
                    title_lens.append(t + 1)
        
        return titles, title_lens

    def init_hidden(self, bsz):
        return Variable(torch.zeros(bsz, self.num_chars).type(self.dtype))

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
