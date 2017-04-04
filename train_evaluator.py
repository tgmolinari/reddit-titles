# Training loop for the evaluator network
import json
import numpy as np
import torch
import torch.autograd as autograd
import torch.utils.data as data
import torchvision.transforms as transforms
from evaluator_net import ScorePredictor
from dataset import PostFolder

# train will pull in the model, expected model is ScorePredictor
# ScorePredictor will contain the feature extractor net from evaluator_net.py

def train(model, args : dict):
	'''
	args -  img_dir : directory the images live in
		    posts_json : filename for the posts json 
            dtype : torch Tensor type
            epochs : number of times to iterate over the training data

    model - evaluator network model
    '''
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
        batch_size=32, shuffle=True,
        num_workers=8, pin_memory=True)

	for epoch in args.epochs:
		total_loss = 0
		for i, (image, title, title_lens, score) in enumerate(train_loader):
			# because torch doesn't have an argsort, we use numpy's
			title_lens = np.array(title_lens.tolist())
			sorted_idx = np.array(np.argsort(title_lens)[::-1])
			sorted_title_lens = title_lens[sorted_idx]

			image_var = autograd.Variable(image[torch.from_numpy(sorted_idx)]).type(args.dtype)
			title_var = autograd.Variable(title[torch.from_numpy(sorted_idx)]).type(args.dtype)
			score_var = autograd.Variable(score[torch.from_numpy(sorted_idx)]).type(args.dtype)
			packed_seq = pack_padded_sequence(title_var, sorted_title_lens, batch_first = True)
			
			loss_val = model.forward(image_var, packed_seq, score_var)
			total_loss += loss_val
			model.backward(loss_val)
			
		if epoch % 3  == 0:
			print("Epoch %d:  %.6f" % (epoch, total_loss/len(posts_json)))
	    