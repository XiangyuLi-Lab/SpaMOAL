import torch.multiprocessing as mp
import torch.distributed as dist
import argparse
import torch
import torch.nn as nn
import numpy as np
np.random.seed(0)
from ruamel.yaml import YAML
import os
from models import DMG,clustering
os.environ['R_HOME'] = '/home/lixiangyu/anaconda3/envs/dmg/lib/R'

def get_args(yaml_path=None) -> argparse.Namespace:
    yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "args.yaml")
    # custom_key = custom_key.split("+")[0]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='E11_0') # input file's name should be dataset_ADT.csv/dataset_Expression_HVG3000.csv/dataset_groundtruth.csv...
    parser.add_argument("--model",default='none')
    parser.add_argument("--omics1",default='RNA', help='name of omics, should be one of "RNA, ADT, ATAC"') # RNA---transcriptom
    parser.add_argument("--omics2",default='none', help='name of omics, should be one of "RNA, ADT, ATAC"') # ADT---protein
    parser.add_argument("--omics3",default='none', help='name of omics, should be one of "RNA, ADT, ATAC"') # ATAC
    parser.add_argument("--omics4",default='none', help='name of omics, should be one of "RNA, ADT, ATAC"') # ADT---protein
    parser.add_argument("--omics5",default='none', help='name of omics, should be one of "RNA, ADT, ATAC"') # ATAC
    parser.add_argument("--num_view",type=int,default=2) # number of modals
    parser.add_argument("--input_folder",default='/input') # where the input files are
    parser.add_argument('--feature_drop', type=float, default=0.01, help='dropout of features') # 0.1
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument("--alpha", default=0.5, help="Reconstruction error coefficient", type=float) # 0.08
    parser.add_argument("--beta", default=0.8, help="Independence constraint coefficient", type=float) # 1
    parser.add_argument("--lammbda", default=0.5, help="Contrastive constraint coefficient", type=float) # 1
    parser.add_argument("--num_iters", default=10, help="Number of training iterations", type=int) # 200
    parser.add_argument("--n_clusters",default=6) # number of clusters
    parser.add_argument("--distance",default='none')
    parser.add_argument('--seed', type=int, default=0, help='the seed to use')
    parser.add_argument("--custom-key", default='Node')
    parser.add_argument('--sc', type=float, default=3.0, help='GCN self connection')
    parser.add_argument('--sparse', type=bool, default=False, help='sparse adjacency matrix') # False
    parser.add_argument('--iterater', type=int, default=10, help='iterater for evaluate')
    parser.add_argument('--use_pretrain', type=bool, default=False, help='use_pretrain') # hyy
    parser.add_argument('--isBias', type=bool, default=False, help='isBias')
    parser.add_argument('--activation', nargs='?', default='relu')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
    parser.add_argument('--gpu_num', type=int, default=0, help='the id of gpu to use')
    parser.add_argument('--test_epo', type=int, default=50, help='test_epo')
    parser.add_argument('--test_lr', type=int, default=0.3, help='test_lr')
    parser.add_argument('--save_root', type=str, default="./saved_model", help='root for saving the model')
    parser.add_argument("--c_dim", default=8, help="Dimensionality of c", type=int) # 8
    parser.add_argument("--p_dim", default=2, help="Dimensionality of p", type=int) # 2
    parser.add_argument("--projector_dim", default=64, help="Dimensionality of projector for clustering", type=int) 
    parser.add_argument("--lr_max", default=1e0, help="Learning rate for maximization", type=float)
    parser.add_argument("--lr_min", default=1e-3, help="Learning rate for minimization", type=float)
    parser.add_argument("--weight_decay", default=1e-4, help="Weight decay for parameters eta", type=float)
    parser.add_argument("--inner_epochs", default=10, help="Number of inner epochs", type=int)
    parser.add_argument("--phi_num_layers", default=2, help="Number of layers for phi", type=int)
    parser.add_argument("--phi_hidden_size", default=256, help="Number of hidden neurons for phi", type=int) # 256
    parser.add_argument("--hid_units", default=256, help="Number of hidden neurons", type=int) # 256
    parser.add_argument("--decolayer", default=2, help="Number of decoder layers", type=int)
    parser.add_argument("--neighbor_num", default=300, help="Number of all sampled neighbor", type=int)
    parser.add_argument("--sample_neighbor", default=30, help="Number of sampled neighbor during each iteration", type=int) # 30
    parser.add_argument("--sample_num", default=50, help="Number of sampled edges during each iteration", type=int) # 50
    parser.add_argument("--tau", default=0.5, help="temperature in contrastive loss", type=float)
    parser.add_argument("--omics_dim", default=3000, help="omics max dimension", type=int)
    parser.add_argument("--adj_neighbor", default=6, help="adj matrix neighbors number", type=int)


    args = parser.parse_args()
    return args

import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA

def printConfig(args):
    arg2value = {}
    for arg in vars(args):
        arg2value[arg] = getattr(args, arg)
    print(arg2value)
    
if __name__ == '__main__':
    args = get_args()
    embedder = DMG(args)
    printConfig(embedder.args)
    embeddings = embedder.training()

    
