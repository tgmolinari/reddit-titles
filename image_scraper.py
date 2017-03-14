import json
import pickle
import os
import requests
import shutil
from multiprocessing import Process, Queue
post_dir = '../reddit_pics/'
img_dir = '../reddit_titles_pics/'

def grab_img(post_queue):
	while True:
		post = post_queue.get()
		if post == None:
			break
		try:
			if '/a/' not in post['url']:
				if 'imgur' in post['url'] and 'com' in post['url'].split('.')[-1]:
					post['url'] += '.jpg'
				resp = requests.get(post['url'])
				if resp.status_code == 200:
					with open(img_dir + post['id'] + '.' + resp.headers['content-type'].split('/')[-1], 'wb') as img_file:
						img_file.write(resp.content)
				else:
					print('Status of ' + str(resp.status_code) + ' on post ' + post['id'])
		except Exception as e:
			print(e)
			continue

			
if __name__ == '__main__':
	grabbed_ids = pickle.load(open("id_set.p","rb"))
# Multiprocessing to pull the images faster
	post_queue = Queue()	
	post_files = os.listdir(post_dir)
# our JSON from the webscraper
	for post_json in post_files:
		posts = json.load(open(post_dir + post_json,"r"))
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
