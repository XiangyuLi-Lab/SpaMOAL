import numba
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import scipy.io as sio
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import OneHotEncoder
import torch as th
import torch.nn.functional as F
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset,WikiCSDataset
from dgl.data import AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset
from dgl.data import CoauthorCSDataset, CoauthorPhysicsDataset, CoraFullDataset
from typing import Optional
from typing import Optional, Tuple, Union
import torch
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch import Tensor
import torch
import concurrent.futures
from sklearn.preprocessing import StandardScaler

# 读入数据
@numba.njit("f4(f4[:], f4[:])")
def euclid_dist(t1,t2):
	sum = 0
	for i in range(t1.shape[0]):
		sum += (t1[i] - t2[i]) ** 2
	return np.sqrt(sum)

@numba.njit("f4[:,:](f4[:,:])", parallel=True, nogil=True)
def pairwise_distance(X):
	n = X.shape[0]
	adj = np.empty((n, n), dtype = np.float32)
	for i in numba.prange(n):
		for j in numba.prange(n):
			adj[i][j] = euclid_dist(X[i], X[j])
	return adj

def calculate_adj_matrix(x,y):
    print("Calculateing adj matrix using xy only...")
    X = np.array([x, y]).T.astype(np.float32)
    return pairwise_distance(X)

def common_adj(input_folder, n=6,dataset=None):
    print("neighbor number:",n)
    spatial = pd.read_csv(input_folder + dataset +"_groundtruth.csv")
    x_coord = spatial['array_row']
    y_coord = spatial['array_col']
    common_adj = calculate_adj_matrix(x = x_coord, y = y_coord)
    for i in range(len(common_adj)):
        min_indices = np.argpartition(common_adj[i], n)[:n]
        common_adj[i] = 0  # 首先将整行置0
        common_adj[i, min_indices] = 1
    print("common_adj.shape: ",common_adj.shape)
    return common_adj

def private_mod1_adj(input_folder, dataset, neighbor=6):
    rna_exp = pd.read_csv(input_folder+dataset+"_Expression_HVG3000.csv",
                          sep=",", na_filter=False, index_col=0)
    # print(rna_exp.head())
    # rna_exp = rna_exp.T
    rna_matrix = cosine_distances(rna_exp)
    for i in range(len(rna_matrix)):
        max_indices = np.argpartition(-rna_matrix[i], neighbor)[:neighbor]
        rna_matrix[i] = 0  # 首先将整行置0
        rna_matrix[i, max_indices] = 1
    print("rna_matrix shape: ",rna_matrix.shape)
    return rna_matrix

def private_mod2_adj(input_folder, neighbor=6):
    atac_exp = pd.read_csv(input_folder+dataset+"_GeneScoreMatrix_HVG3000.csv",
                          sep=",", na_filter=False, index_col=0)
    # print(atac_exp.head())
    # atac_exp = atac_exp.T
    atac_matrix = cosine_distances(atac_exp)
    for i in range(len(atac_matrix)):
        max_indices = np.argpartition(-atac_matrix[i], neighbor)[:neighbor]
        atac_matrix[i] = 0  # 首先将整行置0
        atac_matrix[i, max_indices] = 1
    print("atac_matrix shape:",atac_matrix.shape)
    return atac_matrix


def process_omics(omics, omics_dim,input_folder, dataset, model, distance):
    """
    处理单个omics类型的数据
    """
    if omics == "RNA":
        f1 = input_folder + dataset + "_Expression_HVG3000.csv"
        truefeatures = pd.read_csv(f1, sep=",", na_filter=False, index_col=0)
        print("Expression:", truefeatures.shape)
        n, x = truefeatures.shape
        if x < omics_dim:
            input_tensor = torch.tensor(truefeatures.values, dtype=torch.float32)
            linear_layer = nn.Linear(x, omics_dim, bias=False)
            with torch.no_grad():
                linear_layer.weight.data[:x, :x] = torch.eye(x)
            with torch.no_grad():
                output_tensor = linear_layer(input_tensor)
            truefeatures = pd.DataFrame(output_tensor.detach().numpy(), columns=[f'new_col{i+1}' for i in range(omics_dim)])
            print("升维后矩阵维度:", truefeatures.shape)
        # scaler = StandardScaler()
        # truefeatures = scaler.fit_transform(truefeatures)
        truefeatures = sp.lil_matrix(truefeatures)
        return truefeatures
    elif omics == "ADT":
        truefeatures = pd.read_csv(input_folder + dataset + "_ADT.csv", sep=",", na_filter=False, index_col=0)
        print("ADT:", truefeatures.shape)
        n, x = truefeatures.shape
        if x < omics_dim:
            input_tensor = torch.tensor(truefeatures.values, dtype=torch.float32)
            linear_layer = nn.Linear(x, omics_dim, bias=False)
            with torch.no_grad():
                linear_layer.weight.data[:x, :x] = torch.eye(x)
            with torch.no_grad():
                output_tensor = linear_layer(input_tensor)
            truefeatures = pd.DataFrame(output_tensor.detach().numpy(), columns=[f'new_col{i+1}' for i in range(omics_dim)])
            print("升维后矩阵维度:", truefeatures.shape)
        # scaler = StandardScaler()
        # truefeatures = scaler.fit_transform(truefeatures)
        truefeatures = sp.lil_matrix(truefeatures)
        return truefeatures
    elif omics == "Peak":
        truefeatures = pd.read_csv(input_folder + dataset + "_Peak_HVG3000.csv", sep=",", na_filter=False, index_col=0)
        print("Peak:", truefeatures.shape)
        n, x = truefeatures.shape
        if x < omics_dim:
            input_tensor = torch.tensor(truefeatures.values, dtype=torch.float32)
            linear_layer = nn.Linear(x, omics_dim, bias=False)
            with torch.no_grad():
                linear_layer.weight.data[:x, :x] = torch.eye(x)
            with torch.no_grad():
                output_tensor = linear_layer(input_tensor)
            truefeatures = pd.DataFrame(output_tensor.detach().numpy(), columns=[f'new_col{i+1}' for i in range(omics_dim)])
            print("升维后矩阵维度:", truefeatures.shape)
        # scaler = StandardScaler()
        # truefeatures = scaler.fit_transform(truefeatures)
        truefeatures = sp.lil_matrix(truefeatures)
        return truefeatures
    elif omics == "image_normalize":
        truefeatures = pd.read_csv(input_folder + dataset + "_image_" + model + "_" + distance + "_normalize.csv", sep=",", na_filter=False, index_col=0)
        print("image:", truefeatures.shape)
        n, x = truefeatures.shape
        if x < omics_dim:
            input_tensor = torch.tensor(truefeatures.values, dtype=torch.float32)
            linear_layer = nn.Linear(x, omics_dim, bias=False)
            with torch.no_grad():
                linear_layer.weight.data[:x, :x] = torch.eye(x)
            with torch.no_grad():
                output_tensor = linear_layer(input_tensor)
            truefeatures = pd.DataFrame(output_tensor.detach().numpy(), columns=[f'new_col{i+1}' for i in range(omics_dim)])
            print("升维后矩阵维度:", truefeatures.shape)
        # scaler = StandardScaler()
        # truefeatures = scaler.fit_transform(truefeatures)
        truefeatures = sp.lil_matrix(truefeatures)
        return truefeatures
    elif omics == "image_unnormalize":
        truefeatures = pd.read_csv(input_folder + dataset + "_image_" + model + "_" + distance + "_unnormalize.csv", sep=",", na_filter=False, index_col=0)
        print("image:", truefeatures.shape)
        n, x = truefeatures.shape
        if x < omics_dim:
            input_tensor = torch.tensor(truefeatures.values, dtype=torch.float32)
            linear_layer = nn.Linear(x, omics_dim, bias=False)
            with torch.no_grad():
                linear_layer.weight.data[:x, :x] = torch.eye(x)
            with torch.no_grad():
                output_tensor = linear_layer(input_tensor)
            truefeatures = pd.DataFrame(output_tensor.detach().numpy(), columns=[f'new_col{i+1}' for i in range(omics_dim)])
            print("升维后矩阵维度:", truefeatures.shape)
        # scaler = StandardScaler()
        # truefeatures = scaler.fit_transform(truefeatures)
        truefeatures = sp.lil_matrix(truefeatures)
        return truefeatures
    elif omics == "ATAC":
        truefeatures = pd.read_csv(input_folder + dataset + "_GeneScoreMatrix_HVG3000.csv", sep=",", na_filter=False, index_col=0)
        print("ATAC_GeneScoreMatrix_HVG3000:", truefeatures.shape)
        n, x = truefeatures.shape
        if x < omics_dim:
            input_tensor = torch.tensor(truefeatures.values, dtype=torch.float32)
            linear_layer = nn.Linear(x, omics_dim, bias=False)
            with torch.no_grad():
                linear_layer.weight.data[:x, :x] = torch.eye(x)
            with torch.no_grad():
                output_tensor = linear_layer(input_tensor)
            truefeatures = pd.DataFrame(output_tensor.detach().numpy(), columns=[f'new_col{i+1}' for i in range(omics_dim)])
            print("升维后矩阵维度:", truefeatures.shape)
        # scaler = StandardScaler()
        # truefeatures = scaler.fit_transform(truefeatures)
        truefeatures = sp.lil_matrix(truefeatures)
        return truefeatures
    elif omics == "none":
        return None
    else:
        truefeatures = pd.read_csv(input_folder + dataset + "_"+omics+"_HVG3000.csv", sep=",", na_filter=False, index_col=0)
        print(omics+"_HVG3000.csv", truefeatures.shape)
        n, x = truefeatures.shape
        if x < omics_dim:
            input_tensor = torch.tensor(truefeatures.values, dtype=torch.float32)
            linear_layer = nn.Linear(x, omics_dim, bias=False)
            with torch.no_grad():
                linear_layer.weight.data[:x, :x] = torch.eye(x)
            with torch.no_grad():
                output_tensor = linear_layer(input_tensor)
            truefeatures = pd.DataFrame(output_tensor.detach().numpy(), columns=[f'new_col{i+1}' for i in range(omics_dim)])
            print("升维后矩阵维度:", truefeatures.shape)
        # scaler = StandardScaler()
        # truefeatures = scaler.fit_transform(truefeatures)
        truefeatures = sp.lil_matrix(truefeatures)
        return truefeatures

def load_10x(input_folder, omics_list,omics_dim, sc=3, dataset=None, model="none", omics=None, distance=None, adj_neighbor=6):
    truefeatures_list = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # 提交每个omics类型的处理任务到进程池
        futures = [executor.submit(process_omics, omics,omics_dim, input_folder, dataset, model, distance) for omics in omics_list]
        # 获取每个任务的结果
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                truefeatures_list.append(result)

    common_adj_list = common_adj(input_folder, adj_neighbor, dataset)
    num_node = common_adj_list.shape[0]
    adj_list = []
    label = []
    adj_fusion1 = sp.csr_matrix(common_adj_list)
    for _ in truefeatures_list:
        adj_list.append(adj_fusion1)
    idx_train = idx_val = idx_test = np.arange(len(adj_list))
    return adj_list, truefeatures_list, label, idx_train, idx_val, idx_test, adj_fusion1


def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    print("labels_onehot:-----------------")
    print(labels_onehot)

    return labels_onehot


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    # 计算每行元素的和
    rowsum = features.sum(1)
    
    # 计算每行和的倒数，将无穷大值设为 0
    r_inv = np.zeros(rowsum.shape, dtype=np.float32)
    np.divide(1, rowsum, out=r_inv, where=rowsum!=0)
    
    # 构建稀疏对角矩阵
    r_mat_inv = sp.spdiags(r_inv.flatten(), 0, features.shape[0], features.shape[0])
    
    # 用对角矩阵与原矩阵相乘实现行归一化
    features = r_mat_inv.dot(features)
    
    # 将稀疏矩阵转换为稠密矩阵
    return features.todense()
    
# def preprocess_features(features):
#     """Row-normalize feature matrix and convert to tuple representation"""
#     rowsum = np.array(features.sum(1),dtype=np.float32)
#     r_inv = np.power(rowsum, -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0.
#     r_mat_inv = sp.diags(r_inv)
#     features = r_mat_inv.dot(features)
#     return features.todense()

def normalize_graph(A):
    eps = 2.2204e-16
    deg_inv_sqrt = (A.sum(dim=-1).clamp(min=0.) + eps).pow(-0.5)
    if A.size()[0] != A.size()[1]:
        A = deg_inv_sqrt.unsqueeze(-1) * (deg_inv_sqrt.unsqueeze(-1) * A)
    else:
        A = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

def add_random_edge(edge_index, p: float, force_undirected: bool = False,
                    num_nodes: Optional[Union[Tuple[int], int]] = None,
                    training: bool = True):
    if p < 0. or p > 1.:
        raise ValueError(f'Ratio of added edges has to be between 0 and 1 '
                         f'(got {p}')
    if force_undirected and isinstance(num_nodes, (tuple, list)):
        raise RuntimeError('`force_undirected` is not supported for'
                           ' heterogeneous graphs')

    device = edge_index.device
    if not training or p == 0.0:
        edge_index_to_add = torch.tensor([[], []], device=device)
        return edge_index, edge_index_to_add

    if not isinstance(num_nodes, (tuple, list)):
        num_nodes = (num_nodes, num_nodes)
    num_src_nodes = maybe_num_nodes(edge_index, num_nodes[0])
    num_dst_nodes = maybe_num_nodes(edge_index, num_nodes[1])

    num_edges_to_add = round(edge_index.size(1) * p)
    row = torch.randint(0, num_src_nodes, size=(num_edges_to_add, ))
    col = torch.randint(0, num_dst_nodes, size=(num_edges_to_add, ))

    if force_undirected:
        mask = row < col
        row, col = row[mask], col[mask]
        row, col = torch.cat([row, col]), torch.cat([col, row])
    edge_index_to_add = torch.stack([row, col], dim=0).to(device)
    edge_index = torch.cat([edge_index, edge_index_to_add], dim=1)
    return edge_index, edge_index_to_add

def dropout_edge(edge_index: Tensor, p: float = 0.5,
                 force_undirected: bool = False,
                 training: bool = True) -> Tuple[Tensor, Tensor]:

    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask

    row, col = edge_index

    edge_mask = torch.rand(row.size(0), device=edge_index.device) >= p

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1),),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, drop_prob, isBias=False):
        super(GCN, self).__init__()

        self.fc_1 = nn.Linear(in_ft, out_ft, bias=False)
        if act == 'prelu':
            self.act = nn.PReLU()
        elif act == 'relu':
            self.act = nn.ReLU()
        if isBias:
            self.bias_1 = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias_1.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)
        for m in self.modules():
            self.weights_init(m)
        self.drop_prob = drop_prob
        self.isBias = isBias


    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            #torch.nn.init.xavier_uniform_(m.weight.data)
            torch.nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq = F.dropout(seq, self.drop_prob, training=self.training)
        seq_raw = self.fc_1(seq)
        if sparse:
            seq = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_raw, 0)), 0)
        else:
            seq = torch.mm(adj, seq_raw)

        if self.isBias:
            seq += self.bias_1

        return self.act(seq)


def update_S(model, features, adj_list, c_dim, device):
    model.eval()

    FF = []
    with torch.no_grad():
            # Forward
        common, _ = model.encode(features, adj_list)
        FF.append(torch.cat(common, 1))

        FF = torch.cat(FF, 0)

        # The projection step, i.e., subtract the mean
        FF = FF - torch.mean(FF, 0, True)

        h=[]
        for i in range(2):
            h.append(FF[:,i*c_dim:(i+1)*c_dim])

        FF = torch.stack(h, dim=2)

        # The SVD step
        U, _, T = torch.svd(torch.sum(FF, dim=2))
        S = torch.mm(U, T.t())
        S = S*(FF.shape[0])**0.5
    return S

class Linearlayer(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(Linearlayer, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)

if __name__ == '__main__':
    load_10x()
