import os
import json
import re
import pickle
import nltk
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

# maximum title length in words
max_len = 100
embedding_dims = 50
glove_vectors = pickle.load(open('../glove_vectors_50d.p', 'rb'))

def default_loader(path):
    return Image.open(path).convert('RGB')

def make_dataset(posts_json : list, img_dir, file_ext):
    # Posts json is a list of reddit posts, stored in dictionaries
    # img_dir is the directory housing the downloaded images
    # file_ext is the data type they're stored in
    posts_list = []
    for post in posts_json:
        path = os.path.join(img_dir, post['id'] + file_ext)
        score = 1.0/(1 + np.exp(-2*(post['score'] - 2.5)))
        item = (path, post['title'], score)
        posts_list.append(item)
    return posts_list
       
def default_title_transform(title):
    tokens = nltk.word_tokenize(title)
    # join <URL> like tokens again
    title = []
    while len(tokens) > 0:
        if tokens[0] == '<' and tokens[2] == '>':
            title.append(tokens.pop(0) + tokens.pop(0) + tokens.pop(0))
        else:
            title.append(tokens.pop(0))
    # look up GloVe vectors for each token
    transformed_title = torch.zeros((max_len, embedding_dims))
    i = 0
    while len(title) > 0:
        word = title.pop(0)
        if word in glove_vectors:
            transformed_title[i] = glove_vectors[word].clone()
            i += 1
    return transformed_title, max(i, 1)

def titles_from_padded(padded_seq, title_lens):
    new_titles = [''] * padded_seq.size(0)
    for i in range(padded_seq.size(0)):
        for j in range(title_lens[i]):
            charid = np.argmax(padded_seq[i][j].numpy())
            if padded_seq[i][j][charid] == 1:
                new_titles[i] += id2char[charid]
    return [''.join(title) for title in new_titles]

class PostFolder(data.Dataset):
    def __init__(self, post_json, img_dir, file_ext='.jpeg', transform=None, title_transform=None,
                 loader=default_loader):
        posts = make_dataset(post_json, img_dir, file_ext)
        
        self.img_dir = img_dir
        self.posts = posts
        self.transform = transform
        self.title_transform = title_transform or default_title_transform
        self.loader = loader

    def __getitem__(self, index):
        path, title, score = self.posts[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        title, title_len = self.title_transform(title)
        return img, title, title_len, score

    def __len__(self):
        return len(self.posts)

class ImageFolder(data.Dataset):
    def __init__(self, post_json, img_dir, file_ext='.jpeg', transform=None,
                 loader=default_loader):
        posts = make_dataset(post_json, img_dir, file_ext)
        
        self.img_dir = img_dir
        self.posts = posts
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path, title, score = self.posts[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.posts)