import json
import pickle
import atexit
import os
from urllib.request import urlretrieve
from multiprocessing import Process, Queue

def save_queue(q):
	missed_posts = []
	for i in iter(q.get, 'STOP'):
		missed_posts.append(i)
	json.dump(missed_posts, open("missed_posts.json","w"))

def grab_img(post_queue):
	while True:
		post = post_queue.get()
		if post == None:
			break
		try:
			urlretrieve(post['url'], "/media/tom/shared/reddit_titles_pics/"+post['id'])
		except:
			continue
			# Probably a 404 or 403 we don't care about too much

if __name__ == '__main__':

# Multiprocessing to pull the images faster
	post_queue = Queue()	
	post_files = os.listdir("reddit_pics/")
# our JSON from the webscraper
	for post_json in post_files:
		posts = json.load(open("reddit_pics/"+post_json,"r"))
		for post in posts:
			post_queue.put(post)

# Set up our workers and, should the we have to terminate our grabs early for whatever reason, save the state of the queue

	atexit.register(save_queue, post_queue)
	image_workers = []
	for i in range(30):
		post_queue.put(None)
		worker = Process(target=grab_img, args=(post_queue,))
		worker.start()
		image_workers.append(worker)
	for w in image_workers:
		w.join()
