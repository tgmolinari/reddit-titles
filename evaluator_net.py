# Evaluator network for image, title pairs

import torch
import torch.utils.data
import torchvision.models as models
#from torchvision.models.resnet import ResNet
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#import torch.nn.functional as F
#from torch.autograd import Variable


#def _forward(self, x):
#    x = self.conv1(x)
#    x = self.bn1(x)
#    x = self.relu(x)
#    x = self.maxpool(x)
#
#    x = self.layer1(x)
#    x = self.layer2(x)
#    x = self.layer3(x)
#    x = self.layer4(x)
#
#    return x

feat_extractor = models.vgg16(pretrained=False)#, batch_norm=True)

# Monkeywrenching to decaptitate a pretrained net
#fType = type(feat_extractor.forward)
#feat_extractor.forward = fType(_forward,feat_extractor)


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
	input_var = torch.autograd.Variable(iinput)
	target_var = torch.autograd.Variable(target)
	out = feat_extractor.features(input_var)
	print(out)
