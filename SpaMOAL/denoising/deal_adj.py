import numba
import pandas as pd
import numpy as np
import scanpy as sc
import warnings
import argparse
import time
warnings.filterwarnings("ignore")
# 读入数据
@numba.njit("f4(f4[:], f4[:])")
def euclid_dist(t1,t2):
	sum=0
	for i in range(t1.shape[0]):
		sum+=(t1[i]-t2[i])**2
	return np.sqrt(sum)

@numba.njit("f4[:,:](f4[:,:])", parallel=True, nogil=True)
def pairwise_distance(X):
	n=X.shape[0]
	adj=np.empty((n, n), dtype=np.float32)
	for i in numba.prange(n):
		for j in numba.prange(n):
			adj[i][j]=euclid_dist(X[i], X[j])
	return adj

def calculate_adj_matrix(x,y):
    print("Calculateing adj matrix using xy only...")
    X = np.array([x, y]).T.astype(np.float32)
    return pairwise_distance(X)

def compute_matrix(adj,neighbor):
	n = adj.shape[0]
	# print(n)
	matrix = np.empty((n+1, n+1), dtype=np.float32)
	raw = adj.copy()
	for i in range(n):
		limit = np.sort(raw[i,])[neighbor] # ST(4) VISIUM(6)
		# print(limit)
		for j in range(n):
			if adj[i,j] <= limit and i!=j:
				matrix[i+1,j+1] = 1
			else:
				matrix[i+1,j+1] = 0
	print(matrix.shape)
	return matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", type=str, default="E11_0")
    
    args = parser.parse_args()

    start = time.time()
    spatial = pd.read_csv(f'../Data/{args.name}/{args.name}_groundtruth.csv',sep=",", header=None, na_filter=False, index_col=0)
    spatial = spatial[1:]
    print(spatial)
    x_array = spatial[3].tolist()
    y_array = spatial[4].tolist()
    adj = calculate_adj_matrix(x=x_array, y=y_array)
    matrix = compute_matrix(adj,6)
    
    np.savetxt(f'../Data/{args.name}/cor_{args.name}_position.csv', adj, delimiter=',')
    np.savetxt(f'../Data/{args.name}/adj_{args.name}_position.csv', matrix, delimiter=',')
   
    print("save sucessful!!")
    end = time.time()
    print("run time: "+ str(end-start) + " seconds")

