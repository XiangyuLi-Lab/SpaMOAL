import torch
import torch_geometric.utils
from utils import process
import numpy as np
from torch_geometric.utils import degree,remove_self_loops
from scipy.sparse import coo_matrix
from sklearn.preprocessing import StandardScaler


class embedder:
    def __init__(self, args):
        if args.gpu_num == -1:
            args.device = 'cpu'
        else:
            args.device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")
        
        dataset = args.dataset
        model = args.model
        distance = args.distance
        input_folder = args.input_folder
        omics_dim=args.omics_dim
        omics_list = [args.omics1, args.omics2, args.omics3,args.omics4,args.omics5]
        adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_10x(input_folder, omics_list,omics_dim,args.sc, dataset= dataset, model=model, distance= distance,adj_neighbor=args.adj_neighbor)
        for i in range(args.num_view):
            features[i] = process.preprocess_features(features[i])
        adj_list = [process.sparse_mx_to_torch_sparse_tensor(adj) for adj in adj_list]
        adj_list = [adj.to_dense() for adj in adj_list]
        idx_p_list = []
        sample_edge_list = []
        for adj in adj_list:
            A_degree = degree(adj.to_sparse()._indices()[0], features[0].shape[0], dtype=torch.int)
            out_node = adj.to_sparse()._indices()[1]
            deg_list_0 = torch.cat([torch.tensor([0], dtype=torch.int), torch.cumsum(A_degree, dim=0)])
            idx_p_list_0 = []
            node_indices = torch.arange(features[0].shape[0])
            for j in range(1, args.neighbor_num + 1):
                random_list = deg_list_0[node_indices] + j % A_degree
                idx_p_0 = out_node[random_list]
                idx_p_list_0.append(idx_p_0)
            idx_p_list.append(idx_p_list_0)
        adj_list = [process.normalize_graph(adj) for adj in adj_list]
        if args.sparse:
            adj_list = [adj.to_sparse() for adj in adj_list]
        args.nb_nodes = adj_list[0].shape[0]
        args.nb_classes = labels.shape[1]
        args.ft_size = features[0].shape[1]
        features_list = []
        for i in range(args.num_view):
            features_list.append(features[i])
        self.adj_list = adj_list
        self.features = [torch.FloatTensor(features) for features in features_list]
        print("embedder:",args.device)
        self.labels = torch.FloatTensor(labels).to(args.device)
        self.idx_train = torch.LongTensor(idx_train).to(args.device)
        self.idx_val = torch.LongTensor(idx_val).to(args.device)
        self.idx_test = torch.LongTensor(idx_test).to(args.device)
        self.idx_p_list = idx_p_list
        self.sample_edge_list = sample_edge_list
        self.args = args


