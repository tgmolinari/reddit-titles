# Evaluator network for image, title pairs
import numpy as np
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.model_zoo
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.autograd import Variable

class ScorePredictor(torch.nn.Module):
    def __init__(self, dtype, num_chars, new_cnn=False):
        super(ScorePredictor, self).__init__()

        self.num_chars = num_chars
        self.dtype = dtype
        self.init_cnn(new_cnn)
        self.title_feat_extractor = nn.GRU(self.num_chars, 512,)
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

    def init_cnn(self, new_cnn):
        # Modified forward for VGG so that it acts as a feature extractor
        def _forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier[0](x)
            return x

        self.img_feat_extractor = models.vgg16()

        if new_cnn:
            model_url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
            vgg_state_dict = torch.utils.model_zoo.load_url(model_url)
            self.img_feat_extractor.load_state_dict(vgg_state_dict)

        self.img_feat_extractor.type(self.dtype)

        # Monkeywrenching to decapitate the pretrained net
        fType = type(self.img_feat_extractor.forward)
        self.img_feat_extractor.forward = fType(_forward,self.img_feat_extractor)
        



dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
id2char = {0: 'i', 1: ' ', 2: 't', 3: 'h', 4: 'n', 5: 'k', 6: "'", 7: 'l', 8: 's', 9: 'a', 10: 'y', 11: 'o', 12: 'u', 13: 'd', 14: 'e', 15: 'r', 16: 'v', 17: 'p', 18: 'w', 19: 'm', 20: 'f', 21: 'c', 22: 'b', 23: 'g', 24: 'j', 25: ',', 26: '.', 27: '1', 28: '8', 29: 'x', 30: '2', 31: '4', 32: '!', 33: '?', 34: ':', 35: '(', 36: '5', 37: ')', 38: 'z', 39: '&', 40: ';', 41: '-', 42: 'q', 43: '3', 44: '0', 45: '"', 46: '@', 47: '$', 48: '/', 49: '[', 50: ']', 51: '+', 52: '9', 53: '7', 54: '6', 55: '~', 56: '#', 57: '=', 58: '%', 59: '_', 60: '*', 61: '|', 62: '{', 63: '}', 64: '^', 65: '\\', 66: '`', 67: '\x03', 68: '\x7f', 69: '\x19'}
char2id = {',': 25, '8': 28, 'e': 14, 'g': 23, '{': 62, 'r': 15, ')': 37, '\\': 65, 'b': 22, '_': 59, '+': 51, '.': 26, 'l': 7, 'u': 12, '&': 39, 'a': 9, 'w': 18, ';': 40, ':': 34, '"': 45, 's': 8, '%': 58, '5': 36, '?': 33, '6': 54, '*': 60, ' ': 1, "'": 6, '\x03': 67, '!': 32, 'm': 19, 'k': 5, '2': 30, 'c': 21, '}': 63, '$': 47, 'j': 24, '3': 43, '-': 41, '/': 48, '|': 61, '#': 56, '0': 44, '4': 31, 'y': 10, ']': 50, 'x': 29, 'q': 42, 'z': 38, 'v': 16, 'i': 0, '1': 27, 'h': 3, '7': 53, '^': 64, 'n': 4, '\x7f': 68, '\x19': 69, '=': 57, 'f': 20, 't': 2, '~': 55, '9': 52, '@': 46, 'd': 13, 'o': 11, '(': 35, 'p': 17, '`': 66, '[': 49}
model = ScorePredictor(dtype, len(id2char))
posts = json.load(open('../cleaned_posts.json', 'r+'))

max_len = 400
batch = posts[:32]

title_tensors = torch.zeros(32, max_len, len(id2char))
titles = [post['title'] for post in batch]
title_lens = []
for i, title in enumerate(titles):
    title_lens.append(len(title))
    for j, char in enumerate(title):
        title_tensors[i, j, char2id[char]] = 1

sorted_idx = np.array(np.argsort(title_lens)[::-1])
title_lens = np.array(title_lens)[sorted_idx]
title_tensors = Variable(title_tensors[torch.from_numpy(sorted_idx)])

packed_seq = pack_padded_sequence(title_tensors.type(dtype), title_lens, batch_first = True)

padded_seq, lens = pad_packed_sequence(packed_seq, batch_first = True)


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
    imgs = torch.autograd.Variable(iinput).type(dtype)
    scores = model(imgs, packed_seq)
    print(scores.data)
    break


def titles_from_padded(padded_seq):
    new_titles = [''] * padded_seq.data.size(1)
    for i in range(padded_seq.data.size(0)):
        for j in range(padded_seq.data.size(1)):
            charid = np.argmax(padded_seq.data[i][j].numpy())
            if padded_seq.data[i][j][charid] == 1:
                new_titles[i] += id2char[charid]
    new_titles = [''.join(title) for title in new_titles]
    return new_titles
