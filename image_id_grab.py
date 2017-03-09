import os, pickle
files = os.listdir("/media/tom/shared/reddit_titles_pics/")
grabbed_ids = set()
for file in files:
	grabbed_ids.add(file)
pickle.dump(grabbed_ids, open("id_set.p","wb"))
	