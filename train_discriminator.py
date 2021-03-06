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
from discriminator_net import ImageDiscriminator
from feat_extractor import FeatureExtractor
from dataset import PostFolder
from dataset import titles_from_padded
from tensorboard_logger import configure, log_value

DATASET_SIZE = 681572
NUM_DIMS = 50

# proportion of batch to copy and swap images and titles (MISM_PROP) or mangle titles (MANG_PROP)
MISM_PROP = 0.5
MANG_PROP = 0.5

def train(model, args):
    #set up logger
    timestring = str(date.today()) + '_' + time.strftime("%Hh-%Mm-%Ss", time.localtime(time.time()))
    run_name = 'embedd_discrim_training_r1' + '_' + timestring
    configure("logs/" + run_name, flush_secs=5)

    posts_json = json.load(open(args.posts_json, 'r', encoding='utf-8', errors='replace'))
    print(len(posts_json))
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
            {'params': model.lin1.parameters()}, {'params': model.lin2.parameters()}, 
            {'params': model.attn_lin1.parameters()}, {'params': model.attn_conv1.parameters()}],
            lr=learning_rate)
    # Correctly paired title and image is labeled as 1
    # Mangled titles and mismatched titles are labeled as 0

    bce_loss = nn.BCELoss()

    batch_ctr = 0
    epoch_loss = 0
    for epoch in range(args.epochs):
        if epoch > 0:
            last_epoch_loss = epoch_loss
            epoch_loss = 0 
        for i, (images, titles, title_lens, score) in enumerate(train_loader):
            score = score/score
            # swapping random images and post titles
            num_mism = int(titles.size(0) * MISM_PROP)
            mism_ind = np.random.choice(titles.size(0), num_mism, replace = False)
            # shifting indices by same amount so that a mismatch is ensured
            mism_map = (mism_ind + randint(0, titles.size(0) - 1)) % titles.size(0)
            mism_imgs = images[torch.from_numpy(mism_ind)]
            mism_titles = titles.clone()[torch.from_numpy(mism_map)]
            mism_lens = title_lens[torch.from_numpy(mism_map)]
            
            # mangling titles
            num_mang = int(titles.size(0) * MANG_PROP)
            mang_ind = np.random.choice(titles.size(0), num_mang, replace = False) 
            title_copy = titles.clone()
            num_noise = 100
            noise_tensor = torch.randn(num_noise, NUM_DIMS) * 2
            for ind in mang_ind:
                if title_lens[ind] > 1:
                    num_words_mang = randint(int(np.ceil(title_lens[ind]*.5)), title_lens[ind] - 1)
                    mang_words = np.random.choice(title_lens[ind] - 1, num_words_mang, replace = False)
                    # if randint(0, 1) > 0:
                        # set mangled word to random noise vector
                    # title_copy[ind][torch.from_numpy(mang_words)] = noise_tensor[torch.from_numpy(np.random.choice(num_noise - 1, num_words_mang))]
                    title_copy[ind][torch.from_numpy(mang_words)] = torch.randn(num_words_mang, NUM_DIMS) * 2
                    # else:
                        # randomly change words to other words within title
                        # title_copy[ind][torch.from_numpy(mang_words)] = title_copy[ind][torch.from_numpy(np.random.choice(title_lens[ind] - 1, num_words_mang))]

            mang_imgs = images[torch.from_numpy(mang_ind)]
            mang_titles = title_copy[torch.from_numpy(mang_ind)]
            mang_lens = title_lens[torch.from_numpy(mang_ind)]

            images = torch.cat((images, mism_imgs, mang_imgs), 0)
            titles = torch.cat((titles, mism_titles, mang_titles), 0)
            title_lens = torch.cat((title_lens, mism_lens, mang_lens), 0)
            score = torch.cat((score.type(torch.FloatTensor), torch.zeros(num_mism), torch.zeros(num_mang)), 0)

            images = image_feature_extractor.make_features(Variable(images).type(args.dtype))


            pred_score = model.forward(images, Variable(titles).type(args.dtype), title_lens)

            optimizer.zero_grad()

            score_var = Variable(score).type(args.dtype)
            batch_loss = bce_loss(pred_score, score_var)
            
            log_value('BCE loss', batch_loss.data[0], batch_ctr)
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
                    


parser = argparse.ArgumentParser(description='Discriminator Train')
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

    model = ImageDiscriminator(args.dtype, NUM_DIMS)
    if args.load_name is not None:
        model.load_state_dict(pickle.load(open(args.load_name + '.p', 'rb')))

    train(model, args)