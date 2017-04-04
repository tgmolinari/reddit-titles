# Training loop for the evaluator network
import json
import numpy as np
import time
import argparse
import pickle
from datetime import date

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from evaluator_net import ScorePredictor
from dataset import PostFolder

from tensorboard_logger import configure, log_value

DATASET_SIZE = 682440
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
       		transforms.ToTensor(),
       		normalize,
    	])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
	l1_loss = nn.L1Loss()

	batch_ctr = 0
	for epoch in range(args.epochs):
		for i, (image, title, title_lens, score) in enumerate(train_loader):
			# because torch doesn't have an argsort, we use numpy's
			title_lens = np.array(title_lens.tolist())
			sorted_idx = np.array(np.argsort(title_lens)[::-1])
			sorted_title_lens = title_lens[sorted_idx]

			image_var = autograd.Variable(image[torch.from_numpy(sorted_idx)]).type(args.dtype)
			title_var = autograd.Variable(title[torch.from_numpy(sorted_idx)]).type(args.dtype)
			score_var = autograd.Variable(score[torch.from_numpy(sorted_idx)]).type(args.dtype)
			packed_seq = pack_padded_sequence(title_var, sorted_title_lens, batch_first = True)
			
			pred_score = model.forward(image_var, packed_seq)

	        optimizer.zero_grad()

			batch_loss = l1_loss(pred_score, score_var)

            log_value('L1 log score loss', batch_loss, batch_ctr)
            batch_ctr += 1

			batch_loss.backward()
	        optimizer.step()

            if batch_ctr % 1000 == 0:
                pickle.dump(model.state_dict(), open(args.save_name + '.p', 'wb'))


parser = argparse.ArgumentParser(description='Evaluator Train')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size (default: 32)')
parser.add_argument('--img_dir', type=str,
                    help='image directory')
parser.add_argument('--posts_json', type=str,
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
    
    model = ScorePredictor(args.dtype)

	train(model, args)
	    