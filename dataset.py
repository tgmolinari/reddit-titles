import os
import json
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

# Our predefined char2id and maximum title length
max_len = 400
char2id = {'p': 17, '%': 49, 'b': 22, '!': 32, '?': 33, 'e': 14, 'f': 20, '0': 41, 'o': 11, 's': 8, 'w': 18, 'g': 23, ':': 34, 'k': 5, '7': 46, '8': 28, 'a': 9, 'r': 15, '5': 35, '-': 38, "'": 6, 'z': 36, '*': 51, 'd': 13, 'v': 16, 'i': 0, '/': 44, '#': 48, '&': 37, 'h': 3, '9': 45, 'm': 19, '3': 40, 't': 2, '.': 26, '_': 50, ' ': 1, 'y': 10, 'c': 21, '4': 31, 'n': 4, '6': 47, 'l': 7, '"': 42, '$': 43, 'q': 39, 'j': 24, 'x': 29, 'u': 12, '1': 27, ',': 25, '2': 30}
id2char = {0: 'i', 1: ' ', 2: 't', 3: 'h', 4: 'n', 5: 'k', 6: "'", 7: 'l', 8: 's', 9: 'a', 10: 'y', 11: 'o', 12: 'u', 13: 'd', 14: 'e', 15: 'r', 16: 'v', 17: 'p', 18: 'w', 19: 'm', 20: 'f', 21: 'c', 22: 'b', 23: 'g', 24: 'j', 25: ',', 26: '.', 27: '1', 28: '8', 29: 'x', 30: '2', 31: '4', 32: '!', 33: '?', 34: ':', 35: '5', 36: 'z', 37: '&', 38: '-', 39: 'q', 40: '3', 41: '0', 42: '"', 43: '$', 44: '/', 45: '9', 46: '7', 47: '6', 48: '#', 49: '%', 50: '_', 51: '*'}

def default_loader(path):
    return Image.open(path).convert('RGB')

def make_dataset(posts_json : list, img_dir, file_ext):
    # Posts json is a list of reddit posts, stored in dictionaries
    # img_dir is the directory housing the downloaded images
    # file_ext is the data type they're stored in
    posts_list = []
    for post in posts_json:
        path = os.path.join(img_dir, post['id'] + file_ext)
        item = (path, post['title'], post['log_score'])
        posts_list.append(item)
    return posts_list
       
def default_title_transform(title):
    title_len = len(title)
    transformed_title = torch.zeros((max_len, len(char2id)))
    for i,char in enumerate(title):
        transformed_title[i, char2id[char]] = 1
    return transformed_title, title_len

def titles_from_padded(padded_seq):
    new_titles = [''] * padded_seq.data.size(0)
    for i in range(padded_seq.data.size(0)):
        for j in range(padded_seq.data.size(1)):
            charid = np.argmax(padded_seq.data[i][j].numpy())
            if padded_seq.data[i][j][charid] == 1:
                new_titles[i] += id2char[charid]
    return [''.join(title) for title in new_titles]

class PostFolder(data.Dataset):
    def __init__(self, post_json, img_dir, file_ext='.jpeg', transform=None, title_transform=None,
                 loader=default_loader):
        posts = make_dataset(post_json, img_dir, file_ext)
        if len(posts) == 0:
            raise(RuntimeError("you specified an empty folder"))
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
