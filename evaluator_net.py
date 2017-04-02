# Evaluator network for image, title pairs

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.model_zoo
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class ScorePredictor(torch.nn.Module):
    def __init__(self, num_chars):
        self.num_chars = num_chars
        img_feat_extractor = None
        title_feat_extractor = nn.GRU(self.num_chars, 256, bidirectional=True)
        lin1 = nn.Linear(4096 + 256 * 2, 256)
        lin2 = nn.Linear(256, 1)

    def forward(self, imgs, titles):
        self.init_hidden(titles.size(1))

        img_feat = img_feat_extractor(img)
        title_feat, _ = title_feat_extractor(titles, self.h0)
        # title_feat = title_feat.
        features = torch.cat(title_feat.data, img_feat, 1)
        x = F.elu(lin1(features))
        return lin2(x)

    def init_hidden(self, bsz):
        self.h0 = Variable(torch.zeros(2, bsz, self.num_chars))



# Modified forward for VGG so that it acts as a feature extractor
def _forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier[0](x)
    return x

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
model_url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
vgg_state_dict = torch.utils.model_zoo.load_url(model_url)

feat_extractor = models.vgg16()
feat_extractor.type(dtype)
feat_extractor.load_state_dict(vgg_state_dict)

# Monkeywrenching to decaptitate the pretrained net
fType = type(feat_extractor.forward)
feat_extractor.forward = fType(_forward,feat_extractor)

traindir = "../to_nick/"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(traindir, transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])),
        batch_size=32, shuffle=True,
        num_workers=4, pin_memory=True)

for i, (iinput, target) in enumerate(train_loader):
    input_var = torch.autograd.Variable(iinput).type(dtype)
    target_var = torch.autograd.Variable(target).type(dtype)
    out = feat_extractor.forward(input_var)

