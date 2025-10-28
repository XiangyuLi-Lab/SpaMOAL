import argparse
import numpy
import numpy as np
import warnings
import pandas as pd
import time
import os
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")

def compute_matrix(adj,num):
	n = adj.shape[0]
	matrix = np.empty((n+1, n+1), dtype=np.float32)
	raw = adj.copy()
	for i in range(n):
		limit = np.sort(raw[i,])[num]
		print(limit)
		for j in range(n):
			if adj[i,j] <= limit and i!=j:
				matrix[i+1,j+1] = 1
			else:
				matrix[i+1,j+1] = 0
	print(matrix)
	return matrix

def adj_euclidean(emb):
	n = emb.shape[0]
	print(n)
	print("*"*50)
	adj = np.empty((n, n), dtype=np.float32)
	for i in range(n):
		for j in range(i, n):
			adj[i][j] = adj[j][i] = numpy.sqrt(numpy.sum(numpy.square(emb[i]-emb[j])))
	print("*" * 50)
	print("euclidean:")
	print(adj)
	return adj

def adj_cosine(emb):
	n = emb.shape[0]
	print(n)
	print("*" * 50)
	adj = np.empty((n, n), dtype=np.float32)
	for i in range(n):
		for j in range(i, n):
			adj[i][j] = adj[j][i] = np.dot(emb[i],emb[j])/(np.linalg.norm(emb[i]) * np.linalg.norm(emb[j]))
	print("*" * 50)
	print("cosine:")
	print(adj)
	return adj

def adj_pearson(emb):
	n = emb.shape[0]
	print(n)
	print("*" * 50)
	adj = np.empty((n, n), dtype=np.float32)
	for i in range(n):
		for j in range(i, n):
			i_ = emb[i] - np.mean(i)
			j_ = emb[j] - np.mean(j)
			adj[i][j] = adj[j][i] = np.dot(i_, j_) / (np.linalg.norm(i_) * np.linalg.norm(j_))
	print("*" * 50)
	print("pearson:")
	print(adj)
	return adj

def adj_jaccard(emb):
	n = emb.shape[0]
	print(n)
	print("*" * 50)
	adj = np.empty((n, n), dtype=np.float32)
	for i in range(n):
		for j in range(n):
			i = np.asarray(emb[i], np.int32)
			j = np.asarray(emb[j], np.int32)
			up = np.double(np.bitwise_and((i!=j),np.bitwise_or(i!=0,j!=0)).sum())
			down = np.double(np.bitwise_or(i!=0,j!=0).sum())
			adj[i][j] = up/down
	print("*" * 50)
	print("jaccard:")
	print(adj)
	return adj

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", type=str, default="E11_0")
    parser.add_argument("--model", type=str, default="inception_v3")
    parser.add_argument("--distance", type=str, default="euc") # euclidean cosine pearson jaccard 
    parser.add_argument("--neighbor",type=int,default=6)
    parser.add_argument("--size",type=int,default=299)
    args = parser.parse_args()
   
    start = time.time() 
    emb = np.load(f"../Data/{args.name}/emb_{args.name}_{args.model}_{args.size}.npy")
    #emb = np.load(f"../Data/{args.name}/emb_{args.name}_inception_v3_{args.size}.npy")
    print(emb.shape)
    df2 = pd.DataFrame(emb)
    df2.to_csv(f"../Data/{args.name}/image_feature_{args.name}_{args.model}_PCA.csv", index=False)
    

    model_pca = PCA(n_components=50)
    emb = model_pca.fit_transform(emb)
    print("emb:",emb.shape)
    df2 = pd.DataFrame(emb)
    df2.to_csv(f"../Data/{args.name}/image_feature_{args.name}_{args.model}_PCA.csv", index=False)
    
    if args.distance == "euc":
        adj = adj_euclidean(emb)
    elif args.distance == "cos":
        adj = adj_cosine(emb)
    elif args.distance == "pea":
        adj = adj_pearson(emb)
    elif args.distance =="jac":
        adj = adj_jaccard(emb)
    else:
        adj = adj_euclidean(emb)
    
    
    matrix = compute_matrix(adj,args.neighbor)
    np.savetxt(f'../Data/{args.name}/cor_{args.name}_{args.model}_{args.distance}_{args.size}.csv', adj, delimiter=',')
    np.savetxt(f'../Data/{args.name}/adj_{args.name}_{args.model}_{args.distance}_{args.size}.csv', matrix, delimiter=',')
    
    print("save sucessful!!")
    end = time.time()
    print("run time: "+ str(end-start) + " seconds")
