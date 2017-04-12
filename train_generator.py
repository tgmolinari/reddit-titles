# Training loop for the evaluator network
import json
import numpy as np
import time
import argparse
import pickle
from datetime import date

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from evaluator_net import ScorePredictor
from generator_net import TitleGenerator
from feat_extractor import FeatureExtractor
from dataset import ImageFolder
from dataset import titles_from_padded
from tensorboard_logger import configure, log_value

DATASET_SIZE = 682440
NUM_CHARS = 53

# proportion of batch to copy and swap images and titles (MISM_PROP) or mangle titles (MANG_PROP)
MISM_PROP = 0.25
MANG_PROP = 0.25

# train will pull in the model, expected model is ScorePredictor
# ScorePredictor will contain the feature extractor net from evaluator_net.py

def train(generator, evaluator, args):
    #set up logger
    timestring = str(date.today()) + '_' + time.strftime("%Hh-%Mm-%Ss", time.localtime(time.time()))
    run_name = 'generator_training' + '_' + timestring
    configure("logs/" + run_name, flush_secs=5)

    posts_json = json.load(open(args.posts_json))
    # normalizing function to play nicely with the pretrained feature extractor
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225])
    # specify the dataloader and dataset generator
    train_loader = data.DataLoader(
        ImageFolder(posts_json, args.img_dir, transform=transforms.Compose([
           transforms.RandomHorizontalFlip(),
           transforms.ToTensor(),
           normalize,
           ])),
       batch_size=args.batch_size, shuffle=True,
       num_workers=8, pin_memory=True)

    image_feature_extractor = FeatureExtractor(args.dtype)
    
    learning_rate = 0.001
    optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
    l1_loss = nn.L1Loss()

    batch_ctr = 0
    epoch_loss = 0
    for epoch in range(args.epochs):
        if epoch > 0:
            last_epoch_loss = epoch_loss
        for i, images in enumerate(train_loader):

            image_feats = image_feature_extractor.make_features(Variable(images).type(args.dtype))
            titles, title_lens = generator(image_feats)

            pred_score = evaluator.forward(image_feats, titles, title_lens)
            optimizer.zero_grad()

            batch_loss = -1 * torch.mean(pred_score)
            log_value('Predicted score', torch.mean(pred_score.data), batch_ctr)
            log_value('Learning rate', optimizer.param_groups[0]['lr'], batch_ctr)

            batch_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_loss.data[0]
            batch_ctr += 1
            
            if batch_ctr % 1000 == 0:
                pickle.dump(generator.state_dict(), open(args.save_name + '.p', 'wb'))
        
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

    evaluator = ScorePredictor(args.dtype, NUM_CHARS)
    if args.load_name is not None:
        evaluator.load_state_dict(pickle.load(open(args.load_name + '.p', 'rb')))

    generator = TitleGenerator(args.dtype, NUM_CHARS)

    train(generator, evaluator, args)
