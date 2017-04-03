import os
import json
import torch.utils.data as data
from PIL import Image

def default_loader(path):
    return Image.open(path).convert('RGB')

def make_dataset(posts_json : list, img_dir, file_ext):
    # Posts json is a list of reddit posts, stored in dictionaries
    # img_dir is the directory housing the downloaded images
    # file_ext is the data type they're stored in
    posts_list = []
    for post in posts_json:
        path = os.path.join(img_dir, post['id'] + file_ext)
        item = (path, post['title'], post['score'])
        posts_list.append(item)
    return posts_list

# TODO
# Implement conversion of title strings into our encoded vec
def default_title_transform(title):
    return title

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
        if self.target_transform is not None:
            title = self.title_transform(title)
        return img, title, score

    def __len__(self):
        return len(self.posts)
