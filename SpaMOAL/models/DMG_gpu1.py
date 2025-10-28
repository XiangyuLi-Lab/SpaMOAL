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
device = None
if torch.cuda.is_available():
    device = torch.device("cuda:3")  # 使用GPU
else:
    device = torch.device("cpu")   # 使用CPU



class DMG(embedder):
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
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        features = [feature.to(device) for feature in self.features]
        print(type(features))
        adj_list = [adj.to(device) for adj in self.adj_list]
        print(type(adj_list))

        for i in range(self.args.num_view):
            features[i] = drop_feature(features[i], self.args.feature_drop)

        print("Started training...")
        ae_model = GNNDAE(self.args).to(device)
        predictor =Predictor(self.args.c_dim+self.args.p_dim*2,self.args.projector_dim,self.args.n_clusters).to(device)
        # graph independence regularization network
        mea_func = []
        for i in range(self.args.num_view):
            mea_func.append(Measure_F(self.args.c_dim, self.args.p_dim,
                                  [self.args.phi_hidden_size] * self.args.phi_num_layers,
                                  [self.args.phi_hidden_size] * self.args.phi_num_layers).to(device))
        # Optimizer
        if self.args.num_view == 2:
            optimizer = torch.optim.Adam([
                {'params': mea_func[0].parameters(), 'lr': self.args.lr_max, 'weight_decay': self.args.weight_decay},
                {'params': mea_func[1].parameters(), 'lr': self.args.lr_max, 'weight_decay': self.args.weight_decay},
                {'params': ae_model.parameters(), 'lr': self.args.lr_min}
            ], lr=self.args.lr_min)
        else:
            optimizer = torch.optim.Adam([
                {'params': mea_func[0].parameters(), 'lr': self.args.lr_max, 'weight_decay': self.args.weight_decay},
                {'params': mea_func[1].parameters(), 'lr': self.args.lr_max, 'weight_decay': self.args.weight_decay},
                {'params': mea_func[2].parameters(), 'lr': self.args.lr_max, 'weight_decay': self.args.weight_decay},
                {'params': ae_model.parameters(), 'lr': self.args.lr_min}
            ], lr=self.args.lr_min)

        # model.train()
        ae_model.train()
        mea_func[0].train()
        mea_func[1].train()
        if self.args.num_view == 3:
            mea_func[2].train()
        best = 1e9

        for itr in tqdm(range(1, self.args.num_iters + 1)):

            # Solve the S subproblem
            U = update_S(ae_model, features, adj_list, self.args.c_dim, device)

            # Update network for multiple epochs
            for innerepoch in range(self.args.inner_epochs):
                # Backprop to update
                loss, match_err, recons, corr, contrastive, self_entropy, DDC,common, private = trainmultiplex(ae_model,predictor, mea_func, U, features, adj_list, self.idx_p_list, self.args, optimizer, device, itr*innerepoch)
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
        embedding = []
        hf = update_S(ae_model, features, adj_list, self.args.c_dim, device)
        _, private = ae_model.embed(features, adj_list)
        #private = sum(private) / self.args.num_view
        private = torch.cat(private, dim=1)
        embedding.append(hf)
        embedding.append(private)
        embeddings = torch.cat(embedding, 1)
        projection, cluster_output,probility,predictions=predictor(embeddings)
        predictions_cpu = predictions.cpu()
        data_gd = pd.read_csv(self.args.input_folder+self.args.dataset+"_groundtruth.csv")
        label_encoder = LabelEncoder()
        encoded_ground_truth = label_encoder.fit_transform(data_gd['Ground_Truth'])
        metrics1 = compute_supervised_metrics(encoded_ground_truth, predictions_cpu)
        print('predictions:')
        print(metrics1)
        a = pd.DataFrame(embeddings.cpu().numpy())
        output_path = './output_dmg'
        dataset_folder = f"{output_path}/{self.args.model}_{self.args.tau}_{self.args.omics3}"
        a.to_csv(f"/home/lixiangyu/DMG/DMG-main/benchmark/{self.args.dataset}/{dataset_folder}/{self.args.model}_{self.args.dataset}_{self.args.alpha}_{self.args.beta}_{self.args.lammbda}_{self.args.num_iters}_mclust_DMG_embedding.csv")
        return a



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
    
    # 计算轮廓系数，需要确保数据是numpy数组形式，范围在[-1, 1]，值越大聚类效果越好
    silhouette = silhouette_score(np.array(true_labels).reshape(-1, 1), np.array(predicted_labels).reshape(-1, 1))

    # 计算Calinski-Harabasz指数（CH指数），值越大聚类效果越好
    ch_index = calinski_harabasz_score(np.array(true_labels).reshape(-1, 1), np.array(predicted_labels).reshape(-1, 1))

    # 计算Davies-Bouldin指数（DB指数），值越小聚类效果越好
    db_index = davies_bouldin_score(np.array(true_labels).reshape(-1, 1), np.array(predicted_labels).reshape(-1, 1))

    return {
        'AMI': ami,
        'NMI': nmi,
        'ARI': ari,
        'Homogeneity': homogeneity,
        'V-measure': v_measure,
        'Mutual Information': mutual_info,
        'Silhouette Coefficient': silhouette,
        'Calinski-Harabasz Index': ch_index,
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
        if use_pca: 
            adata = mclust_R(adata, used_obsm=key + '_pca', num_cluster=n_clusters)
        else:
            adata = mclust_R(adata, used_obsm=key, num_cluster=n_clusters)
        adata.obs[add_key] = adata.obs['mclust']
        print("metrics:mclust====================================================")
        metrics2 = compute_supervised_metrics(data['Ground_Truth'], adata.obs['mclust'])
        print(metrics2)
    elif method == 'leiden':
        if use_pca: 
            res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
        else:
            res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment) 
        sc.tl.leiden(adata, random_state=0, resolution=res)
        adata.obs[add_key] = adata.obs['leiden']
        print("metrics:leiden====================================================")
        metrics3 = compute_supervised_metrics(data['Ground_Truth'], adata.obs['leiden'])
        print(metrics3)
    elif method == 'louvain':
        # if use_pca: 
        #     res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
        # else:
        #     res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment) 
        # sc.tl.louvain(adata, random_state=0, resolution=res)
        # adata.obs[add_key] = adata.obs['louvain']
        cluster_louvain(adata, dataset,n_clusters)
        print("metrics:louvain====================================================")
        metrics1 = compute_supervised_metrics(data['Ground_Truth'], adata.obs['louvain'])
        print(metrics1)
    df1 = pd.DataFrame(adata.obsm['X_umap'], columns=['DMG_uamp1', 'DMG_uamp2'])
    df2 = pd.DataFrame(adata.obs)

    output_path = './output_dmg'
    dataset_folder = f"{output_path}/{dataset}"
    os.makedirs(dataset_folder, exist_ok=True)
    if model == 'none':
        embedding.to_csv(f"{dataset_folder}/{model}_{dataset}_{alpha}_{beta}_{lammbda}_{num_iters}_{method}_DMG_embedding.csv")
        df1.to_csv(f"{dataset_folder}/{model}_{dataset}_{alpha}_{beta}_{lammbda}_{num_iters}_{method}_DMG_umap.csv")
        df2.to_csv(f"{dataset_folder}/{model}_{dataset}_{alpha}_{beta}_{lammbda}_{num_iters}_{method}_DMG_label.csv")
    else:
        embedding.to_csv(f"{dataset_folder}/{model}_{dataset}_{distance}_{alpha}_{beta}_{lammbda}_{num_iters}_{method}_DMG_embedding.csv")
        df1.to_csv(f"{dataset_folder}/{model}_{dataset}_{distance}_{alpha}_{beta}_{lammbda}_{num_iters}_{method}_DMG_umap.csv")
        df2.to_csv(f"{dataset_folder}/{model}_{dataset}_{distance}_{alpha}_{beta}_{lammbda}_{num_iters}_{method}_DMG_label.csv")

    return adata




def compute_corr(x1, x2):
    # Subtract the mean
    x1_mean = torch.mean(x1, 0, True)
    x1 = x1 - x1_mean
    x2_mean = torch.mean(x2, 0, True)
    x2 = x2 - x2_mean

    # Compute the cross correlation
    sigma1 = torch.sqrt(torch.mean(x1.pow(2)))
    sigma2 = torch.sqrt(torch.mean(x2.pow(2)))
    corr = torch.abs(torch.mean(x1*x2))/(sigma1*sigma2)

    return corr

# The loss function for matching and reconstruction
def loss_matching_recons(s, x_hat, x, U_batch, idx_p_list, args, epoch):
    l = torch.nn.MSELoss(reduction='sum')

    # Matching loss
    match_err = l(torch.cat(s, 1), U_batch.repeat(1, args.num_view))/s[0].shape[0]
    recons_err = 0
    # Feature reconstruction loss
    for i in range(args.num_view):
        recons_err += l(x_hat[i], x[i])
    recons_err /= s[0].shape[0]

    # Topology reconstruction loss
    interval = int(args.neighbor_num/args.sample_neighbor)
    neighbor_embedding = []
    for i in range(args.num_view):
        neighbor_embedding_0 = []
        for j in range(0, args.sample_neighbor+1):
            neighbor_embedding_0.append(x[i][idx_p_list[i][(epoch + interval * j) % args.neighbor_num]])
        neighbor_embedding.append(sum(neighbor_embedding_0) / args.sample_neighbor)
    recons_nei = 0
    for i in range(args.num_view):
        recons_nei += l(x_hat[i], neighbor_embedding[i])
    recons_nei /= s[0].shape[0]

    return match_err, recons_err, recons_nei

# The loss function for independence regularization
def loss_independence(phi_c_list, psi_p_list):
    # Correlation
    corr = 0
    for i in range(len(phi_c_list)):
        corr += compute_corr(phi_c_list[i], psi_p_list[i])
    return corr

# Contrastive loss
def loss_contrastive(U, private, adj_list, predictions,args):
    i = 0
    loss = 0
    for adj in adj_list:
        adj = adj_list[i]
        out_node = adj.to_sparse()._indices()[1]
        random = np.random.randint(out_node.shape[0], size=int((out_node.shape[0] / args.sample_num)))
        sample_edge = adj.to_sparse()._indices().T[random]
        positive_idx = []
        negative_idx = []
        for idx, edge in enumerate(sample_edge):
            node1, node2 = edge[0], edge[1]
            if predictions[node1] == predictions[node2]:
                positive_idx.append(idx)
            else:
                negative_idx.append(idx)
        positive_sample_edge = sample_edge[positive_idx]
        negative_sample_edge = sample_edge[negative_idx]
        private_sample_0 = private[i][positive_sample_edge.T[0]]
        private_sample_1 = private[i][positive_sample_edge.T[1]]
        private_sample_2 = private[i][negative_sample_edge.T[0]]
        private_sample_3 = private[i][negative_sample_edge.T[1]]
        i += 1
        loss += semi_loss(private_sample_0, private_sample_1, private_sample_2, private_sample_3, args)
    return loss


def semi_loss(z1, z2, z3, z4, args):
    f = lambda x: torch.exp(x / args.tau)
    positive = f(F.cosine_similarity(z1, z2))
    negative = f(F.cosine_similarity(z3, z4))
    return -torch.log(
        positive.sum()
        / (positive.sum() + negative.sum() ))



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
    # Cauchy-Schwarz divergence.
    nom = torch.t(A) @ K @ A
    dnom_squared = torch.unsqueeze(torch.diagonal(nom), -1) @ torch.unsqueeze(torch.diagonal(nom), 0)
    nom = _atleast_epsilon(nom)
    eps = 1e-9
    dnom_squared = _atleast_epsilon(dnom_squared, eps=eps ** 2)      
    d = (2/ (n_clusters * (n_clusters - 1))* triu(nom / torch.sqrt(dnom_squared)))
    return d
  
def kernel_from_distance_matrix(dist, rel_sigma):
    min_sigma = 1e-9
    dist = F.relu(dist)
    sigma2 = rel_sigma * torch.median(dist)
    # Disable gradient for sigma
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
    
def DDC_loss(hidden,probility,n_clusters):
    n_clusters = int(n_clusters)
    cluster_outputs = probility
    rel_sigma=0.15
    hidden_kernel = kernel_from_distance_matrix(cdist(hidden, hidden), rel_sigma)
    # L_1 loss
    loss = d_cs(cluster_outputs, hidden_kernel, n_clusters)

    # L_3 loss
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

def trainmultiplex(model,predictor, mea_func, U, features, adj_list,idx_p_list, args,  optimizer, device, epoch):
    model.train()
    mea_func[0].train()
    mea_func[1].train()
    if args.num_view == 3:
        mea_func[2].train()
    common, private, recons = model(features, adj_list)
    embedding = []
    privates = torch.cat(private, dim=1)
    embedding.append(U)
    embedding.append(privates)
    embeddings = torch.cat(embedding, 1)
    projection, cluster_output,probility,predictions=predictor(embeddings)
    self_entropy = self_entropy_loss(probility)
    DDC = DDC_loss(projection,probility,args.n_clusters)
    match_err, recons_err,recons_nei = loss_matching_recons(common, recons, features, U, idx_p_list, args, epoch)
    # Independence regularizer loss
    phi_c_list = []
    psi_p_list = []
    for i in range(args.num_view):
        phi_c, psi_p = mea_func[i](common[i], private[i])
        phi_c_list.append(phi_c)
        psi_p_list.append(psi_p)
    corr = loss_independence(phi_c_list, psi_p_list)
    data_gd = pd.read_csv("/home/lixiangyu/DMG/DMG-main/input/human_lymph_node_A1/human_lymph_node_A1_groundtruth.csv")
    label_encoder = LabelEncoder()
    encoded_ground_truth = label_encoder.fit_transform(data_gd['Ground_Truth'])
    loss_con = loss_contrastive(U, private, adj_list,encoded_ground_truth, args)
    # Compute the overall loss, note that we use the gradient reversal trick
    # and that's why we have a 'minus' for the last term
    #loss = match_err + args.alpha*recons_err - args.beta* corr + args.lammbda * loss_con
    loss = match_err +  args.alpha*(recons_err+recons_nei) - args.beta* corr + args.lammbda * loss_con + self_entropy + DDC

    # Update all the parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, match_err, recons_err , corr, loss_con, self_entropy, DDC, common, private

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

def grad_reverse(x, coeff):
    return GradientReversalLayer.apply(x, coeff)


class GNNEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pipe = GCN(args.ft_size, args.hid_units, args.activation, args.dropout, args.isBias)
        # map to common
        self.S = nn.Linear(args.hid_units, args.c_dim)
        # map to private
        self.P = nn.Linear(args.hid_units, args.p_dim)

    def forward(self, x, adj):
        tmp = self.pipe(x, adj)
        common = self.S(tmp)
        private = self.P(tmp)
        return common, private


class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.linear1 = Linearlayer(args.decolayer,args.c_dim + args.p_dim, args.hid_units, args.ft_size)
        self.linear2 = nn.Linear(args.ft_size, args.ft_size)

    def forward(self, s, p):
        
        recons = self.linear1(torch.cat((s, p), 1))
        recons = self.linear2(F.relu(recons))
        return recons

class GNNDAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_view = self.args.num_view
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for _ in range(self.args.num_view):
            self.encoder.append(GNNEncoder(args))
            self.decoder.append(Decoder(args))

    def encode(self, x, adj_list):
        common = []
        private = []
        for i in range(self.args.num_view):
            tmp = self.encoder[i](x[i], adj_list[i])
            common.append(tmp[0])
            private.append(tmp[1])

        return common, private

    def decode(self, s, p):
        recons = []
        for i in range(self.num_view):
            tmp = self.decoder[i](s[i], p[i])
            recons.append(tmp)

        return recons

    def forward(self, x, adj):
        common, private = self.encode(x, adj)
        recons = self.decode(common, private)

        return common, private, recons

    def embed(self, x, adj_list):
        common = []
        private = []
        for i in range(self.args.num_view):
            tmp = self.encoder[i](x[i], adj_list[i])
            common.append(tmp[0].detach())
            private.append(tmp[1].detach())
        return common, private

class MLP(nn.Module):
    def __init__(self, input_d, structure, output_d, dropprob=0.0):
        super(MLP, self).__init__()
        self.net = nn.ModuleList()
        self.dropout = torch.nn.Dropout(dropprob)
        struc = [input_d] + structure + [output_d]

        for i in range(len(struc)-1):
            self.net.append(nn.Linear(struc[i], struc[i+1]))

    def forward(self, x):
        for i in range(len(self.net)-1):
            x = F.relu(self.net[i](x))
            x = self.dropout(x)
        # For the last layer
        y = self.net[-1](x)

        return y

class Predictor(nn.Module):
    def __init__(self, input_d, hidden_d,output_d, dropprob=0.0):
        output_d=int(output_d)
        super().__init__()
        self.projector = nn.Linear(input_d, hidden_d)
        self.relu = nn.ReLU()
        self.cluster = nn.Linear(hidden_d, output_d)
        self.probility = nn.Softmax(dim=1)
        self.weights_init(self.projector)
        self.weights_init(self.cluster)
        

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight.data)
            
    def forward(self, x):
        projection = self.relu(self.projector(x))
        cluster_output = self.cluster(projection)
        probility = self.probility(cluster_output)
        # for net in self.nets:
        #     x=net(x)
        y=torch.argmax(probility, axis=1)    
        return projection,cluster_output,probility,y

#measurable functions \phi and \psi
class Measure_F(nn.Module):
    def __init__(self, view1_dim, view2_dim, phi_size, psi_size, latent_dim=1):
        super(Measure_F, self).__init__()
        self.phi = MLP(view1_dim, phi_size, latent_dim)
        self.psi = MLP(view2_dim, psi_size, latent_dim)
        # gradient reversal layer
        self.grl1 = GradientReversalLayer()
        self.grl2 = GradientReversalLayer()

    def forward(self, x1, x2):
        y1 = self.phi(grad_reverse(x1,1))
        y2 = self.psi(grad_reverse(x2,1))
        return y1, y2
