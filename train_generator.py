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
from generator_net import TitleGenerator
from feat_extractor import FeatureExtractor
from dataset import PostFolder
from tensorboard_logger import configure, log_value

DATASET_SIZE = 682440
NUM_DIMS = 50

# train will pull in the model, expected model is ImageDiscriminator

def train(generator, args):
    #set up logger
    timestring = str(date.today()) + '_' + time.strftime("%Hh-%Mm-%Ss", time.localtime(time.time()))
    run_name = args.save_name + '_' + timestring
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
    
    learning_rate = 0.0001
    optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
    cos_loss = nn.CosineEmbeddingLoss()
    mse_loss = nn.MSELoss()
    cos_ones = Variable(torch.ones(args.batch_size)).type(args.dtype)
    embed_zeros = Variable(torch.zeros(NUM_DIMS)).type(args.dtype)
    

    batch_ctr = 0
    epoch_loss = 0
    for epoch in range(args.epochs):
        if epoch > 0:
            last_epoch_loss = epoch_loss
        for i, (images, titles, title_lens, score) in enumerate(train_loader):

            image_feats = image_feature_extractor.make_features(Variable(images).type(args.dtype))
            
            # putting seq_index first, then batch
            titles = torch.transpose(titles, 0, 1)
            titles_v = Variable(titles).type(args.dtype)

            # Train generator on next word
            optimizer.zero_grad()
            gen_pred = generator(image_feats)

            batch_loss = 0
            for t in range(titles.size(0)):
                # if titles_v[t].equal(embed_zeros):
                gen_loss = mse_loss(gen_pred[t], titles_v[t])
                # else:
                #     gen_loss = cos_loss(titles_v[t], gen_pred[t], cos_ones)
                batch_loss += gen_loss.data[0]
                gen_loss.backward(retain_variables=True)
                
            optimizer.step()
            
            log_value('Generator cosine loss', batch_loss / args.batch_size, batch_ctr)
            log_value('Learning rate', optimizer.param_groups[0]['lr'], batch_ctr)

            epoch_loss += gen_loss.data[0]
            batch_ctr += 1
            
            if batch_ctr % 10000 == 0:
                pickle.dump(generator.state_dict(), open(args.save_name + '.p', 'wb'))

        if epoch > 1: #arbitrary epoch choice 
            if (last_epoch_loss - epoch_loss)/last_epoch_loss < .005:
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

    generator = TitleGenerator(args.dtype, NUM_DIMS)
    if args.load_name is not None:
        generator.load_state_dict(pickle.load(open(args.load_name + '.p', 'rb')))

    train(generator, args)
