import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo
import torchvision.models as models

class FeatureExtractor(object):
    def __init__(self, dtype):
        self.dtype = dtype
        # Modified forward for VGG so that it acts as a feature extractor
        self.model = models.vgg16()
        model_url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
        vgg_state_dict = torch.utils.model_zoo.load_url(model_url)
        self.model.type(self.dtype)
        
        self.model.load_state_dict(vgg_state_dict)
        def _forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier[0](x)
            return x

        # want to ensure that our autograd Variables don't backprop to the feat extractor
        for param in self.model.parameters():
            param.requires_grad = False
        # Monkeywrenching to decapitate the pretrained net
        fType = type(self.model.forward)
        self.model.forward = fType(_forward,self.model)

    def make_features(self, img_batch):
        img_batch = self.model.forward(img_batch)
        return img_batch
