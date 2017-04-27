import annoy
import pickle
import torch


em = pickle.load(open("../glove_vectors_50d.p","rb"))
id2word = {}
indexer = annoy.AnnoyIndex(50, metric='euclidean')
#indexer.load('100_zero_emb.ann')

for i,k in enumerate(em):
	if len(em[k]) == 50:
		indexer.add_item(i, em[k])
		id2word[i] = k


indexer.build(100)
null = torch.zeros(50)

results = indexer.get_nns_by_vector(null,10,search_k=100000,include_distances = True)
print(results)
for k in results[0]:
	print(id2word[k])

