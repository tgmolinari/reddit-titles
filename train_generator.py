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
from discriminator_net import ImageDiscriminator
from generator_net import TitleGenerator
from feat_extractor import FeatureExtractor
from dataset import PostFolder
from dataset import titles_from_padded
from tensorboard_logger import configure, log_value

DATASET_SIZE = 682440
NUM_DIMS = 50

# train will pull in the model, expected model is ImageDiscriminator

def train(generator, discriminator, args):
    #set up logger
    timestring = str(date.today()) + '_' + time.strftime("%Hh-%Mm-%Ss", time.localtime(time.time()))
    run_name = args.gen_save_name + '_' + timestring
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
    
    learning_rate = 0.0001
    gen_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)

    batch_ctr = 0
    epoch_loss = 0
    for epoch in range(args.epochs):
        if epoch > 0:
            last_epoch_loss = epoch_loss
        for i, (images, titles, title_lens, score) in enumerate(train_loader):
            print(batch_ctr)
            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)


            image_feats = image_feature_extractor.make_features(Variable(images).type(args.dtype))

            # Train discriminator on real and generated titles
            disc_optimizer.zero_grad()
            
            real_pred = discriminator(image_feats, Variable(titles).type(args.dtype))
            gen_titles = generator(Variable(image_feats.data, volatile = True))
            gen_pred = discriminator(image_feats, Variable(gen_titles.data))
            disc_loss = -torch.mean(real_pred - gen_pred)
            disc_loss.backward()

            disc_optimizer.step()

            log_value('WGAN Discriminator loss', disc_loss.data[0], batch_ctr)
            log_value('WGAN Discriminator loss (real)', torch.mean(real_pred.data), batch_ctr)
            log_value('WGAN Discriminator loss (generated)', -torch.mean(gen_pred.data), batch_ctr)
            

            if batch_ctr % 5 == 0:
                # Train generator on loss of discriminator for generated titles
            
                # This causes .backward() to segfault?!??
                # for p in discriminator.parameters():
                #     p.requires_grad = False

                gen_optimizer.zero_grad()
                gen_titles = generator(image_feats)
                gen_pred = discriminator(image_feats, gen_titles)
                gen_loss = -torch.mean(gen_pred)
                gen_loss.backward()
                gen_optimizer.step()
                log_value('WGAN Generator loss', gen_loss.data[0], batch_ctr)
            
            batch_ctr += 1
            if batch_ctr % 1000 == 0:
                pickle.dump(generator.state_dict(), open(args.gen_save_name + '.p', 'wb'))

            if batch_ctr % 1000 == 0:
                pickle.dump(discriminator.state_dict(), open(args.disc_save_name + '.p', 'wb'))
        


parser = argparse.ArgumentParser(description='Evaluator Train')
parser.add_argument('--batch-size', type=int, default=32,
    help='batch size (default: 32)')
parser.add_argument('--img-dir', type=str,
    help='image directory')
parser.add_argument('--posts-json', type=str,
    help='path to json with reddit posts')
parser.add_argument('--gen-save-name', type=str,
    help='file name to save model params dict')
parser.add_argument('--disc-save-name', type=str,
    help='file name to save model params dict')
parser.add_argument('--gen-load-name', type=str,
    help='file name to load model params dict')
parser.add_argument('--disc-load-name', type=str,
    help='file name to load model params dict')
parser.add_argument('--gpu', action="store_true",
    help='attempt to use gpu')
parser.add_argument('--epochs', type=int, default=9999999999999,
    help='number of epochs to train, defaults to run indefinitely')

if __name__ == '__main__':
    args = parser.parse_args()
    args.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() and args.gpu else torch.FloatTensor
    discriminator = ImageDiscriminator(args.dtype, NUM_DIMS)
    if args.disc_load_name is not None:
        discriminator.load_state_dict(pickle.load(open(args.disc_load_name + '.p', 'rb')))

    generator = TitleGenerator(args.dtype, NUM_DIMS)
    if args.gen_load_name is not None:
        discriminator.load_state_dict(pickle.load(open(args.gen_load_name + '.p', 'rb')))

    train(generator, discriminator, args)
