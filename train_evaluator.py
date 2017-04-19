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
       num_workers=4, pin_memory=True)

    image_feature_extractor = FeatureExtractor(args.dtype)
    
    learning_rate = 0.001
    optimizer = optim.Adam([{'params': model.title_feat_extractor.parameters()},
            {'params': model.lin1.parameters()}, {'params': model.lin2.parameters()}], lr=learning_rate)
    l2_loss = nn.MSELoss()

    batch_ctr = 0
    epoch_loss = 0
    for epoch in range(args.epochs):
        if epoch > 0:
            last_epoch_loss = epoch_loss
            epoch_loss = 0 
        for i, (images, titles, title_lens, score) in enumerate(train_loader):

            images = image_feature_extractor.make_features(Variable(images).type(args.dtype))


            pred_score = model.forward(images, Variable(titles).type(args.dtype), title_lens)

            optimizer.zero_grad()

            score_var = Variable(score).type(args.dtype)
            batch_loss = l2_loss(pred_score, score_var)
            if batch_ctr >= 100:
                log_value('L2 transformed score loss', batch_loss.data[0], batch_ctr)
                log_value('Learning rate', optimizer.param_groups[0]['lr'], batch_ctr)
            epoch_loss += batch_loss.data[0]
            batch_ctr += 1

            batch_loss.backward()
            optimizer.step()
            if batch_ctr % 10000 == 0:
                pickle.dump(model.state_dict(), open(args.save_name + '.p', 'wb'))
                
        if epoch > 2: #arbitrary epoch choice 
            if (last_epoch_loss - epoch_loss)/epoch_loss < .003:
                for param in range(len(optimizer.param_groups)):
                    optimizer.param_groups[param]['lr'] = optimizer.param_groups[param]['lr']/2



parser = argparse.ArgumentParser(description='Evaluator Train')
parser.add_argument('--batch-size', type=int, default=32,
    help='batch size (default: 32)')
parser.add_argument('--img-dir', type=str,
    help='image directory')
parser.add_argument('--posts-json', type=str,
    help='path to json with reddit posts')
parser.add_argument('--save-name', type=str, default='reddit_evaluator_net',
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