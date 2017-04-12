# Training loop for the evaluator network
import json
import numpy as np
import time
import argparse
import pickle
from datetime import date
from random import randint

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from evaluator_net import ScorePredictor
from feat_extractor import FeatureExtractor
from dataset import PostFolder
from dataset import titles_from_padded
from tensorboard_logger import configure, log_value

DATASET_SIZE = 682440
NUM_CHARS = 53

# proportion of batch to copy and swap images and titles (MISM_PROP) or mangle titles (MANG_PROP)
MISM_PROP = 0.25
MANG_PROP = 0.25

# train will pull in the model, expected model is ScorePredictor
# ScorePredictor will contain the feature extractor net from evaluator_net.py

def train(model, args):
    #set up logger
    timestring = str(date.today()) + '_' + time.strftime("%Hh-%Mm-%Ss", time.localtime(time.time()))
    run_name = 'score_eval_training' + '_' + timestring
    configure("logs/" + run_name, flush_secs=5)

    posts_json = json.load(open(args.posts_json))
    # normalizing function to play nicely with the pretrained feature extractor
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225])
    # specify the dataloader and dataset generator
    train_loader = data.DataLoader(
        PostFolder(posts_json, args.img_dir, transform=transforms.Compose([
           transforms.RandomHorizontalFlip(),
           transforms.ToTensor(),
           normalize,
           ])),
       batch_size=args.batch_size, shuffle=True,
       num_workers=8, pin_memory=True)

    image_feature_extractor = FeatureExtractor(args.dtype)
    
    learning_rate = 0.001
    optimizer = optim.Adam([{'params': model.title_feat_extractor.parameters()},
            {'params': model.lin1.parameters()}, {'params': model.lin2.parameters()}], lr=learning_rate)
    l1_loss = nn.L1Loss()

    batch_ctr = 0
    epoch_loss = 0
    for epoch in range(args.epochs):
        if epoch > 0:
            last_epoch_loss = epoch_loss
        for i, (images, titles, title_lens, score) in enumerate(train_loader):

            # swapping random images and post titles
            num_mism = int(titles.size(0) * MISM_PROP)
            mism_ind = np.random.choice(titles.size(0), num_mism, replace = False)
            mism_map = (mism_ind + randint(0, titles.size(0) - 1)) % titles.size(0)
            mism_imgs = images[torch.from_numpy(mism_ind)]
            mism_titles = titles.clone()[torch.from_numpy(mism_map)]
            mism_lens = title_lens[torch.from_numpy(mism_map)]

            # mangling titles
            num_mang = int(titles.size(0) * MANG_PROP)
            mang_ind = np.random.choice(titles.size(0), num_mang, replace = False) 
            title_copy = titles.clone()
            chars_tensor = torch.eye(NUM_CHARS)
            for ind in mang_ind:
                num_chars_title = randint(1+int(title_lens[ind]*.1), title_lens[ind] - 1)
                mang_chars = np.random.choice(title_lens[ind] - 1, num_chars_title, replace = False)
                if randint(0, 1) > 0:
                    # uniformly random character substitution
                    title_copy[ind][torch.from_numpy(mang_chars)] = chars_tensor[torch.from_numpy(np.random.choice(NUM_CHARS - 1, num_chars_title))]
                else:
                    # randomly change characters to other characters within title
                    title_copy[ind][torch.from_numpy(mang_chars)] = title_copy[ind][torch.from_numpy(np.random.choice(title_lens[ind] - 1, num_chars_title))]

            mang_imgs = images[torch.from_numpy(mang_ind)]
            mang_titles = title_copy[torch.from_numpy(mang_ind)]
            mang_lens = title_lens[torch.from_numpy(mang_ind)]

            images = torch.cat((images, mism_imgs, mang_imgs), 0)
            titles = torch.cat((titles, mism_titles, mang_titles), 0)
            title_lens = torch.cat((title_lens, mism_lens, mang_lens), 0)
            score = torch.cat((score.type(torch.FloatTensor), torch.ones(num_mism + num_mang) * -1), 0)

            images = image_feature_extractor.make_features(Variable(images).type(args.dtype))


            pred_score = model.forward(images, Variable(titles).type(args.dtype), title_lens)

            optimizer.zero_grad()

            score_var = Variable(score).type(args.dtype)
            batch_loss = l1_loss(pred_score, score_var)
            log_value('L1 log score loss', batch_loss.data[0], batch_ctr)
            log_value('Learning rate', optimizer.param_groups[0]['lr'], batch_ctr)
            epoch_loss += batch_loss.data[0]
            batch_ctr += 1

            batch_loss.backward()
            optimizer.step()
            if batch_ctr % 1000 == 0:
                pickle.dump(model.state_dict(), open(args.save_name + '.p', 'wb'))
                
        if epoch > 2: #arbitrary epoch choice 
            if -.003 < (last_epoch_loss - epoch_loss)/epoch_loss < .003:
                for param in range(len(optimizer.param_groups)):
                    optimizer.param_groups[param]['lr'] = optimizer.param_groups[param]['lr']/2



parser = argparse.ArgumentParser(description='Evaluator Train')
parser.add_argument('--batch-size', type=int, default=32,
    help='batch size (default: 32)')
parser.add_argument('--img-dir', type=str,
    help='image directory')
parser.add_argument('--posts-json', type=str,
    help='path to json with reddit posts')
parser.add_argument('--save-name', type=str,
    help='file name to save model params dict')
parser.add_argument('--load-name', type=str,
    help='file name to load model params dict')
parser.add_argument('--gpu', action="store_true",
    help='attempt to use gpu')
parser.add_argument('--epochs', type=int, default=9999999999999,
    help='number of epochs to train, defaults to run indefinitely')

if __name__ == '__main__':
    args = parser.parse_args()
    args.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() and args.gpu else torch.FloatTensor

    model = ScorePredictor(args.dtype, NUM_CHARS)
    if args.load_name is not None:
        model.load_state_dict(pickle.load(open(args.load_name + '.p', 'rb')))

    train(model, args)