import os
import scanpy as sc
import pandas as pd
import torch.nn as nn
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np
from tqdm import tqdm
import random as random
import torch
from typing import Any, Optional, Tuple
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as AMI, normalized_mutual_info_score as NMI, adjusted_rand_score as ARI, homogeneity_score, v_measure_score, mutual_info_score
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix
import sys
def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=0):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    print(used_obsm)
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    print(res)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def search_res(adata, n_clusters, use_rep, method='leiden', start=0.1, end=3.0, increment=0.01):
    """\
    Search for the best resolution for leiden or louvain to match the desired number of clusters.
    """
    for resolution in np.arange(start, end, increment):
        if method == 'leiden':
            sc.tl.leiden(adata, resolution=resolution, random_state=0, key_added='leiden_temp')
            num_clusters = adata.obs['leiden_temp'].nunique()
            if num_clusters == n_clusters:
                print(f"Found optimal resolution: {resolution}")
                return resolution
        elif method == 'louvain':
            sc.tl.louvain(adata, resolution=resolution, random_state=0, key_added='louvain_temp')
            num_clusters = adata.obs['louvain_temp'].nunique()
            if num_clusters == n_clusters:
                print(f"Found optimal resolution: {resolution}")
                return resolution
    return resolution  # If no exact match, return the last value of resolution


def cluster_louvain(adata, dataset,n_clusters):
    
    current_clusters = -1  # 初始化当前聚类数
    resolution = 0.5  # 初始化分辨率参数

    itemAdd = 1
    while current_clusters != n_clusters:
        if itemAdd == 50:
            break
        sc.tl.louvain(adata, resolution=resolution, key_added="louvain")
        current_clusters = adata.obs['louvain'].nunique()
        print("current_clusters: ",current_clusters)
        itemAdd += 1
        if current_clusters < n_clusters: # 根据当前聚类数和目标聚类数调整分辨率参数
            resolution += 0.01  # 增加分辨率
        elif current_clusters > n_clusters:
            resolution -= 0.01  # 减小分辨率


# 计算监督指标
def compute_supervised_metrics(true_labels, predicted_labels):
    # 计算各指标
    ami = AMI(true_labels, predicted_labels)
    nmi = NMI(true_labels, predicted_labels)
    ari = ARI(true_labels, predicted_labels)
    homogeneity = homogeneity_score(true_labels, predicted_labels)
    v_measure = v_measure_score(true_labels, predicted_labels)
    mutual_info = mutual_info_score(true_labels, predicted_labels)
    
    return {
        'AMI': ami,
        'NMI': nmi,
        'ARI': ari,
        'Homogeneity': homogeneity,
        'V-measure': v_measure,
        'Mutual Information': mutual_info
    }
    
def clustering(embedding, dataset, alpha, beta, lammbda, num_iters,input_folder, model='none',distance='none', n_clusters=7, key='X', add_key='cluster_result', method='mclust', start=0.1, end=3.0, increment=0.01, use_pca=False, n_comps=20,seed=0):


    print("embedding:", embedding.shape)
    cellinfo = pd.DataFrame(embedding.index, index=embedding.index, columns=['sample_index'])
    geneinfo = pd.DataFrame(embedding.columns, index=embedding.columns, columns=['genes_index'])
    adata = sc.AnnData(csr_matrix(embedding), obs=cellinfo, var=geneinfo)
    adata.var_names_make_unique()
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    
    if method == 'mclust':
        if use_pca: 
            adata = mclust_R(adata, used_obsm=key + '_pca', num_cluster=n_clusters,random_seed=seed)
        else:
            adata = mclust_R(adata, used_obsm=key, num_cluster=n_clusters,random_seed=seed)
        adata.obs[add_key] = adata.obs['mclust']

    elif method == 'leiden':
        if use_pca: 
            res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
        else:
            res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment) 
        sc.tl.leiden(adata, random_state=0, resolution=res)
        adata.obs[add_key] = adata.obs['leiden']
    elif method == 'louvain':
        cluster_louvain(adata, dataset,n_clusters)
    df1 = pd.DataFrame(adata.obsm['X_umap'], columns=['DMG_uamp1', 'DMG_uamp2'])
    df2 = pd.DataFrame(adata.obs)

    output_path = './output'
    dataset_folder = f"{output_path}/{dataset}"
    os.makedirs(dataset_folder, exist_ok=True)
    if model == 'none':
        df1.to_csv(f"{dataset_folder}/{dataset}_{alpha}_{beta}_{lammbda}_{num_iters}_{method}_{n_clusters}_umap.csv")
        df2.to_csv(f"{dataset_folder}/{dataset}_{alpha}_{beta}_{lammbda}_{num_iters}_{method}_{n_clusters}_label.csv")
    else:
        df1.to_csv(f"{dataset_folder}/{model}_{dataset}_{distance}_{alpha}_{beta}_{lammbda}_{num_iters}_{method}_{n_clusters}_umap.csv")
        df2.to_csv(f"{dataset_folder}/{model}_{dataset}_{distance}_{alpha}_{beta}_{lammbda}_{num_iters}_{method}_{n_clusters}_label.csv")
    return adata
    
    
    


