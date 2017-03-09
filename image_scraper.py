import json
import pickle
import os
import urllib.request
from multiprocessing import Process, Queue

def grab_img(post_queue):
	while True:
		post = post_queue.get()
		if post == None:
			break
		try:
			urllib.request.urlretrieve(post['url'], "/media/tom/shared/reddit_titles_pics/"+post['id'])
		except Exception as e:
			print(e)
			continue

			

if __name__ == '__main__':
	grabbed_ids = pickle.load(open("id_set.p","rb"))
# Multiprocessing to pull the images faster
	post_queue = Queue()	
	post_files = os.listdir("reddit_pics/")
# our JSON from the webscraper
	for post_json in post_files:
		posts = json.load(open("reddit_pics/"+post_json,"r"))
		for post in posts:
			if post['id'] in grabbed_ids:
				continue
			else:
				post_queue.put(post)

# Set up our workers
	image_workers = []
	for i in range(30):
		post_queue.put(None)
		worker = Process(target=grab_img, args=(post_queue,))
		worker.start()
		image_workers.append(worker)
	for w in image_workers:
		w.join()
