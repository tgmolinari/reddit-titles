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
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence
from evaluator_net import ScorePredictor
from dataset import PostFolder
from dataset import titles_from_padded
from tensorboard_logger import configure, log_value

DATASET_SIZE = 682440
NUM_CHARS = 52

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
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    l1_loss = nn.L1Loss()

    batch_ctr = 0
    epoch_loss = 0
    for epoch in range(args.epochs):
        if epoch > 0:
            last_epoch_loss = epoch_loss
        for i, (image, title, title_lens, score) in enumerate(train_loader):

            # swapping random images and post titles
            num_mism = int(title.size(0) * MISM_PROP)
            mism_ind = np.random.choice(title.size(0), num_mism, replace = False)
            mism_map = (mism_ind + randint(0, title.size(0) - 1)) % title.size(0)
            mism_imgs = image[torch.from_numpy(mism_ind)]
            mism_titles = title.clone()[torch.from_numpy(mism_map)]
            mism_lens = title_lens[torch.from_numpy(mism_map)]

            # mangling titles
            num_mang = int(title.size(0) * MANG_PROP)
            mang_ind = np.random.choice(title.size(0), num_mang, replace = False) 
            title_copy = title.clone()
            chars_tensor = torch.eye(NUM_CHARS)
            for ind in mang_ind:
                num_chars_title = randint(1+int(title_lens[ind]*.1), title_lens[ind])
                mang_chars = np.random.choice(title_lens[ind], num_chars_title, replace = False)
                if randint(0, 1) > 0:
                    # uniformly random character substitution
                    title_copy[ind][torch.from_numpy(mang_chars)] = chars_tensor[torch.from_numpy(np.random.choice(NUM_CHARS, num_chars_title))]
                else:
                    # randomly change characters to other characters within title
                    title_copy[ind][torch.from_numpy(mang_chars)] = title_copy[ind][torch.from_numpy(np.random.choice(title_lens[ind], num_chars_title))]

            mang_imgs = image[torch.from_numpy(mang_ind)]
            mang_titles = title_copy[torch.from_numpy(mang_ind)]
            mang_lens = title_lens[torch.from_numpy(mang_ind)]

            image = torch.cat((image, mism_imgs, mang_imgs), 0)
            title = torch.cat((title, mism_titles, mang_titles), 0)
            title_lens = torch.cat((title_lens, mism_lens, mang_lens), 0)
            score = torch.cat((score.type(torch.FloatTensor), torch.ones(num_mism + num_mang) * -1), 0)

            # because torch doesn't have an argsort, we use numpy's
            sorted_idx = np.array(np.argsort(title_lens.numpy())[::-1])
            sorted_title_lens = title_lens.numpy()[sorted_idx]

            image_var = autograd.Variable(image[torch.from_numpy(sorted_idx)]).type(args.dtype)
            title_var = autograd.Variable(title[torch.from_numpy(sorted_idx)]).type(args.dtype)
            score_var = autograd.Variable(score[torch.from_numpy(sorted_idx)]).type(args.dtype)
            packed_seq = pack_padded_sequence(title_var, sorted_title_lens, batch_first = True)

            pred_score = model.forward(image_var, packed_seq)

            optimizer.zero_grad()

            batch_loss = l1_loss(pred_score, score_var)
            log_value('L1 log score loss', batch_loss.data[0], batch_ctr)
            log_value('Learning rate', optimizer.param_groups[0]['lr'], batch_ctr)
            epoch_loss += batch_loss.data[0]
            batch_ctr += 1

            batch_loss.backward()
            optimizer.step()

            if batch_ctr % 1000 == 0:
                pickle.dump(model.state_dict(), open(args.save_name + '.p', 'wb'))
        
        if epoch > 3: #arbitrary epoch choice 
            if last_epoch_loss/epoch_loss < .0005:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/2



parser = argparse.ArgumentParser(description='Evaluator Train')
parser.add_argument('--batch-size', type=int, default=32,
    help='batch size (default: 32)')
parser.add_argument('--img-dir', type=str,
    help='image directory')
parser.add_argument('--posts-json', type=str,
    help='path to json with reddit posts')
parser.add_argument('--save_name', type=str,
    help='file name to save model params dict')
parser.add_argument('--gpu', action="store_true",
    help='attempt to use gpu')
parser.add_argument('--epochs', type=int, default=9999999999999,
    help='number of epochs to train, defaults to run indefinitely')

if __name__ == '__main__':
    args = parser.parse_args()
    args.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() and args.gpu else torch.FloatTensor

    model = ScorePredictor(args.dtype, NUM_CHARS)

    train(model, args)
