import os
import scanpy as sc
import pandas as pd
import torch.nn as nn
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import sys
import os
import torch
import random
import numpy as np
from evaluate import evaluate
from embedder import embedder
from utils.process import GCN, update_S, drop_feature, Linearlayer
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
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import LabelEncoder
import torch.multiprocessing as mp
import torch.distributed as dist


torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)

class DMG_s(embedder):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.criteria = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()
        self.log_sigmoid = nn.LogSigmoid()
        if not os.path.exists(self.args.save_root):
            os.makedirs(self.args.save_root)

    def get_params(self, deep=True):
        return {'args':self.args,'alpha': self.args.alpha, 'beta': self.args.beta,'lammbda':self.args.lammbda}

    def set_params(self,alpha,beta,lammbda):
        args=self.args
        args.alpha=alpha
        args.beta=beta
        args.lammbda=lammbda
        self.args=args

    def training(self):
        seed = self.args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = self.args.device
        
        features = self.features[0].to(device)
        adj = self.adj_list[0].to(device)
        features = drop_feature(features, self.args.feature_drop)

        print("Started training...")
        ae_model = GNNDAE(self.args).to(device)
        # 单模态：输入 = 公有特征 + 私有特征
        predictor = Predictor(self.args.c_dim + self.args.p_dim, self.args.projector_dim, self.args.n_clusters).to(device)

        # 单模态不需要模态间独立性正则，直接去掉 Measure_F
        optimizer = torch.optim.Adam(ae_model.parameters(), lr=self.args.lr_min)

        ae_model.train()
        best = 1e9
        cnt_wait = 0

        for itr in tqdm(range(1, self.args.num_iters + 1)):
            # 单模态：U 直接取公有表示，不再做模态对齐
            U = ae_model.encode([features], [adj])[0][0]

            for innerepoch in range(self.args.inner_epochs):
                loss, recons, contrastive, self_entropy, DDC, common, private = trainmultiplex(
                    ae_model, predictor, U, features, adj, self.idx_p_list, self.args, optimizer, device, itr*innerepoch
                )

            if loss < best:
                best = loss
                cnt_wait = 0
            elif loss > best and itr > 100:
                cnt_wait += 1
            if cnt_wait == self.args.patience:
                print("Early stopped!")
                break

            print('====> Iteration: {} Loss = {:.4f}'.format(itr, loss))

        if self.args.use_pretrain:
            ae_model.load_state_dict(
                torch.load('saved_model/best_{}_{}.pkl'.format(self.args.dataset, self.args.custom_key)))

        print("Evaluating...")
        ae_model.eval()
        common, private = ae_model.embed([features], [adj])
        embeddings = torch.cat([common[0], private[0]], dim=1)
        projection, cluster_output, probility, predictions = predictor(embeddings)
        
        predictions_cpu = predictions.cpu()
        data_gd = pd.read_csv(self.args.input_folder + self.args.dataset + "_groundtruth.csv")
        label_encoder = LabelEncoder()
        encoded_ground_truth = label_encoder.fit_transform(data_gd['Ground_Truth'])
        # metrics1 = compute_supervised_metrics(encoded_ground_truth, predictions_cpu)
        # print('predictions:')
        # print(metrics1)
        
        a = pd.DataFrame(embeddings.cpu().numpy())
        output_path = './output_dmg'
        dataset_folder = f"{output_path}/{self.args.model}_{self.args.omics3}"

        return a


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=0):
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])
    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def search_res(adata, n_clusters, use_rep, method='leiden', start=0.1, end=3.0, increment=0.01):
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
    return resolution

def cluster_louvain(adata, dataset,n_clusters):
    current_clusters = -1
    resolution = 0.5
    itemAdd = 1
    while current_clusters != n_clusters:
        if itemAdd == 50:
            break
        sc.tl.louvain(adata, resolution=resolution, key_added="louvain")
        current_clusters = adata.obs['louvain'].nunique()
        itemAdd += 1
        if current_clusters < n_clusters:
            resolution += 0.01
        elif current_clusters > n_clusters:
            resolution -= 0.01

def compute_supervised_metrics(true_labels, predicted_labels):
    ami = AMI(true_labels, predicted_labels)
    nmi = NMI(true_labels, predicted_labels)
    ari = ARI(true_labels, predicted_labels)
    homogeneity = homogeneity_score(true_labels, predicted_labels)
    v_measure = v_measure_score(true_labels, predicted_labels)
    mutual_info = mutual_info_score(true_labels, predicted_labels)
    silhouette = silhouette_score(np.array(true_labels).reshape(-1, 1), np.array(predicted_labels).reshape(-1, 1))
    ch_index = calinski_harabasz_score(np.array(true_labels).reshape(-1, 1), np.array(predicted_labels).reshape(-1, 1))
    db_index = davies_bouldin_score(np.array(true_labels).reshape(-1, 1), np.array(predicted_labels).reshape(-1, 1))
    return {
        'AMI': ami, 'NMI': nmi, 'ARI': ari, 'Homogeneity': homogeneity,
        'V-measure': v_measure, 'Mutual Information': mutual_info,
        'Silhouette Coefficient': silhouette, 'Calinski-Harabasz Index': ch_index,
        'Davies-Bouldin Index': db_index
    }

def clustering(embedding, dataset, alpha, beta, lammbda, num_iters,input_folder, model='none',distance='none', n_clusters=7, key='X', add_key='cluster_result', method='mclust', start=0.1, end=3.0, increment=0.01, use_pca=False, n_comps=20):
    data = pd.read_csv(input_folder+dataset+"_groundtruth.csv")
    print("embedding:", embedding.shape)
    cellinfo = pd.DataFrame(embedding.index, index=embedding.index, columns=['sample_index'])
    geneinfo = pd.DataFrame(embedding.columns, index=embedding.columns, columns=['genes_index'])
    adata = sc.AnnData(csr_matrix(embedding), obs=cellinfo, var=geneinfo)
    adata.var_names_make_unique()
    adata.obs['cell_type'] = data['Ground_Truth'].values
    label_encoder = LabelEncoder()
    data['Ground_Truth'] = label_encoder.fit_transform(data['Ground_Truth'])
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    if method == 'mclust':
        adata = mclust_R(adata, used_obsm=key, num_cluster=n_clusters)
        adata.obs[add_key] = adata.obs['mclust']
        print("metrics:mclust====================================================")
        metrics2 = compute_supervised_metrics(data['Ground_Truth'], adata.obs['mclust'])
        print(metrics2)
    elif method == 'leiden':
        res = search_res(adata, n_clusters, use_rep=key, method=method)
        sc.tl.leiden(adata, random_state=0, resolution=res)
        adata.obs[add_key] = adata.obs['leiden']
        print("metrics:leiden====================================================")
        metrics3 = compute_supervised_metrics(data['Ground_Truth'], adata.obs['leiden'])
        print(metrics3)
    elif method == 'louvain':
        cluster_louvain(adata, dataset,n_clusters)
        print("metrics:louvain====================================================")
        metrics1 = compute_supervised_metrics(data['Ground_Truth'], adata.obs['louvain'])
        print(metrics1)
    df1 = pd.DataFrame(adata.obsm['X_umap'], columns=['DMG_uamp1', 'DMG_uamp2'])
    df2 = pd.DataFrame(adata.obs)
    output_path = './output_dmg'
    dataset_folder = f"{output_path}/{dataset}"
    os.makedirs(dataset_folder, exist_ok=True)
    embedding.to_csv(f"{dataset_folder}/{model}_{dataset}_{alpha}_{beta}_{lammbda}_{num_iters}_{method}_DMG_embedding.csv")
    df1.to_csv(f"{dataset_folder}/{model}_{dataset}_{alpha}_{beta}_{lammbda}_{num_iters}_{method}_DMG_umap.csv")
    df2.to_csv(f"{dataset_folder}/{model}_{dataset}_{alpha}_{beta}_{lammbda}_{num_iters}_{method}_DMG_label.csv")
    return adata

def compute_corr(x1, x2):
    x1_mean = torch.mean(x1, 0, True)
    x1 = x1 - x1_mean
    x2_mean = torch.mean(x2, 0, True)
    x2 = x2 - x2_mean
    sigma1 = torch.sqrt(torch.mean(x1.pow(2)))
    sigma2 = torch.sqrt(torch.mean(x2.pow(2)))
    corr = torch.abs(torch.mean(x1*x2))/(sigma1*sigma2)
    return corr

# ========== 单模态损失：移除匹配损失match_err、模态对齐 ==========
def loss_recons(recons, x, idx_p_list, args, epoch):
    l = torch.nn.MSELoss(reduction='sum')
    recons_err = l(recons, x) / x.shape[0]
    interval = int(args.neighbor_num / args.sample_neighbor)
    neighbor_embedding = x[idx_p_list[0][(epoch + interval * 0) % args.neighbor_num]]
    recons_nei = l(recons, neighbor_embedding) / x.shape[0]
    return recons_err, recons_nei

def loss_contrastive(U, private, adj, predictions, args):
    out_node = adj.to_sparse()._indices()[1]
    random_idx = np.random.randint(out_node.shape[0], size=int(out_node.shape[0] / args.sample_num))
    sample_edge = adj.to_sparse()._indices().T[random_idx]
    positive_idx = [i for i, (n1, n2) in enumerate(sample_edge) if predictions[n1] == predictions[n2]]
    negative_idx = [i for i, (n1, n2) in enumerate(sample_edge) if predictions[n1] != predictions[n2]]
    pos = sample_edge[positive_idx]
    neg = sample_edge[negative_idx]
    z1, z2 = private[pos.T[0]], private[pos.T[1]]
    z3, z4 = private[neg.T[0]], private[neg.T[1]]
    return semi_loss(z1, z2, z3, z4, args)

def semi_loss(z1, z2, z3, z4, args):
    f = lambda x: torch.exp(x / args.tau)
    positive = f(F.cosine_similarity(z1, z2))
    negative = f(F.cosine_similarity(z3, z4))
    return -torch.log(positive.sum() / (positive.sum() + negative.sum()))

class BaseLoss:
    eps = 1e-9
    def __init__(self, model):
        self.n_output = len(list(model.clusters[0].parameters())[0])
        self.weight = 1
    @staticmethod
    def compute_distance(is_binary_input, output, target):
        if is_binary_input:
            return F.binary_cross_entropy(output, target)
        else:
            return F.mse_loss(output, target)
            
def triu(X):
    return torch.sum(torch.triu(X, diagonal=1))

def _atleast_epsilon(X, eps=1e-9):
    return torch.where(X < eps, X.new_tensor(eps), X)

def d_cs(A, K, n_clusters):
    nom = torch.t(A) @ K @ A
    dnom_squared = torch.unsqueeze(torch.diagonal(nom), -1) @ torch.unsqueeze(torch.diagonal(nom), 0)
    nom = _atleast_epsilon(nom)
    dnom_squared = _atleast_epsilon(dnom_squared, eps=1e-9**2)      
    d = (2/ (n_clusters * (n_clusters - 1))* triu(nom / torch.sqrt(dnom_squared)))
    return d
  
def kernel_from_distance_matrix(dist, rel_sigma):
    min_sigma = 1e-9
    dist = F.relu(dist)
    sigma2 = rel_sigma * torch.median(dist)
    sigma2 = sigma2.detach()
    sigma2 = torch.where(sigma2 < min_sigma, sigma2.new_tensor(min_sigma), sigma2)
    k = torch.exp(-dist / (2 * sigma2))
    return k

def cdist(X, Y):
    xyT = X @ torch.t(Y)
    x2 = torch.sum(X ** 2, dim=1, keepdim=True)
    y2 = torch.sum(Y ** 2, dim=1, keepdim=True)
    d = x2 - 2 * xyT + torch.t(y2)
    return d
    
def DDC_loss(hidden,probility,n_clusters,device):
    n_clusters = int(n_clusters)
    cluster_outputs = probility
    rel_sigma=0.15
    hidden_kernel = kernel_from_distance_matrix(cdist(hidden, hidden), rel_sigma)
    loss = d_cs(cluster_outputs, hidden_kernel, n_clusters)
    eye=torch.eye(n_clusters).to(device)
    m = torch.exp(-cdist(cluster_outputs, eye))
    loss += d_cs(m, hidden_kernel, n_clusters)
    return loss

def self_entropy_loss(probility):
    eps = 1e-8
    prob_mean = probility.mean(dim=0)
    prob_mean[(prob_mean < eps).data] = eps
    loss = (prob_mean * torch.log(prob_mean)).sum()
    return loss

# ========== 核心训练函数：彻底删除匹配损失、模态对齐、互信息正则 ==========
def trainmultiplex(model,predictor, U, features, adj,idx_p_list, args,  optimizer, device, epoch):
    model.train()
    common, private, recons = model([features], [adj])
    embeddings = torch.cat([common[0], private[0]], dim=1)
    projection, cluster_output, probility, predictions = predictor(embeddings)
    
    self_entropy = self_entropy_loss(probility)
    DDC = DDC_loss(projection, probility, args.n_clusters, device)
    recons_err, recons_nei = loss_recons(recons[0], features, idx_p_list, args, epoch)
    loss_con = loss_contrastive(U, private[0], adj, predictions, args)
    
    # 单模态最终损失：重构 + 对比 + 聚类正则，**无匹配损失、无模态对齐**
    loss = args.alpha*(recons_err+recons_nei) + args.lammbda*loss_con + self_entropy + DDC

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, recons_err, loss_con, self_entropy, DDC, common, private

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, coeff=1.):
        ctx.coeff = coeff
        return input * 1.0
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.coeff, None

def grad_reverse(x, coeff):
    return GradientReversalLayer.apply(x, coeff)

class GNNEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pipe = GCN(args.ft_size, args.hid_units, args.activation, args.dropout, args.isBias)
        self.S = nn.Linear(args.hid_units, args.c_dim)
        self.P = nn.Linear(args.hid_units, args.p_dim)

    def forward(self, x, adj):
        tmp = self.pipe(x, adj)
        return self.S(tmp), self.P(tmp)

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.linear1 = Linearlayer(args.decolayer, args.c_dim + args.p_dim, args.hid_units, args.ft_size)
        self.linear2 = nn.Linear(args.ft_size, args.ft_size)

    def forward(self, s, p):
        recons = self.linear1(torch.cat((s, p), 1))
        recons = self.linear2(F.relu(recons))
        return recons

class GNNDAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = GNNEncoder(args)
        self.decoder = Decoder(args)

    def encode(self, x, adj_list):
        c, p = self.encoder(x[0], adj_list[0])
        return [c], [p]

    def decode(self, s, p):
        return [self.decoder(s[0], p[0])]

    def forward(self, x, adj):
        c, p = self.encode(x, adj)
        r = self.decode(c, p)
        return c, p, r

    def embed(self, x, adj_list):
        c, p = self.encoder(x[0], adj_list[0])
        return [c.detach()], [p.detach()]

class MLP(nn.Module):
    def __init__(self, input_d, structure, output_d, dropprob=0.0):
        super().__init__()
        self.net = nn.ModuleList()
        self.dropout = nn.Dropout(dropprob)
        struc = [input_d] + structure + [output_d]
        for i in range(len(struc)-1):
            self.net.append(nn.Linear(struc[i], struc[i+1]))

    def forward(self, x):
        for i in range(len(self.net)-1):
            x = F.relu(self.net[i](x))
            x = self.dropout(x)
        return self.net[-1](x)

class Predictor(nn.Module):
    def __init__(self, input_d, hidden_d, output_d, dropprob=0.0):
        super().__init__()
        self.projector = nn.Linear(input_d, hidden_d)
        self.relu = nn.ReLU()
        self.cluster = nn.Linear(hidden_d, int(output_d))
        self.probability = nn.Softmax(dim=1)
        torch.nn.init.kaiming_normal_(self.projector.weight.data)
        torch.nn.init.kaiming_normal_(self.cluster.weight.data)

    def forward(self, x):
        projection = self.relu(self.projector(x))
        cluster_output = self.cluster(projection)
        prob = self.probability(cluster_output)
        return projection, cluster_output, prob, torch.argmax(prob, axis=1)