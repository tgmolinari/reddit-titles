# Training loop for the evaluator network

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
		    posts_json : list of json dicts for reddit posts 
            dtype : torch Tensor type
            epochs : number of times to iterate over the training data

    model - evaluator network model
    '''

    # normalizing function to play nicely with the pretrained feature extractor
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
	# specify the dataloader and dataset generator
	train_loader = data.DataLoader(
    	PostFolder(args.posts_json,args.img_dir, transform=transforms.Compose([
       		transforms.ToTensor(),
       		normalize,
    	])),
        batch_size=32, shuffle=True,
        num_workers=8, pin_memory=True)

	for epoch in args.epochs:
		total_loss = 0
		for i, (image, title, score) in enumerate(train_loader):
			image_var = autograd.Variable(image).type(args.dtype)
			title_var = autograd.Variable(title).type(args.dtype)
			score_var = autograd.Variable(score).type(args.dtype)
			loss_val = model.forward(image_var,title_var,score_var)
			total_loss += loss_val
			model.backward(loss_val)
			
		if epoch % 3  == 0:
			print("Epoch %d:  %.6f" % (epoch, total_loss/len(posts_json)))
	    