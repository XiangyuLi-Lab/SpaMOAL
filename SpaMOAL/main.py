import argparse
import torch
import torch.nn as nn
import numpy as np
np.random.seed(0)
from ruamel.yaml import YAML
import os
from models import DMG,clustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
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
    parser.add_argument("--num_view",type=int,default=2) # number of modals
    parser.add_argument("--input_folder",default='/input') # where the input files are
    parser.add_argument('--feature_drop', type=float, default=0.01, help='dropout of features') # 0.1
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument("--alpha", default=0.5, help="Reconstruction error coefficient", type=float) # 0.08
    parser.add_argument("--beta", default=0.5, help="Independence constraint coefficient", type=float) # 1
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
    parser.add_argument('--gpu_num', type=int, default=1, help='the id of gpu to use')
    parser.add_argument('--test_epo', type=int, default=50, help='test_epo')
    parser.add_argument('--test_lr', type=int, default=0.3, help='test_lr')
    parser.add_argument('--save_root', type=str, default="./saved_model", help='root for saving the model')
    parser.add_argument("--c_dim", default=8, help="Dimensionality of c", type=int) # 8
    parser.add_argument("--p_dim", default=2, help="Dimensionality of p", type=int) # 2
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


    # with open(yaml_path) as args_file:
    ##     args = parser.parse_args()
    #     args_key = "-".join([args.model_name, args.dataset, args.custom_key])
    #     try:
    #         parser.set_defaults(**dict(YAML().load(args_file)[args_key].items()))
    #     except KeyError:
    #         raise AssertionError("KeyError: there's no {} in yamls".format(args_key), "red")

    # Update params from .yamls
    args = parser.parse_args()
    return args


# 检查GPU是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda:1")  # 使用GPU
    else:
        device = torch.device("cpu")   # 使用CPU
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA



def printConfig(args):
    arg2value = {}
    for arg in vars(args):
        arg2value[arg] = getattr(args, arg)
    print(arg2value)

class WrappedModel:
    def __init__(self, args,alpha,beta,lammbda):
        # 初始化原始模型，将传入的参数传递给原始模型
        self.args=args
        self.alpha=alpha
        self.beta=beta
        self.lammbda=lammbda
        self.args.alpha=alpha
        self.args.beta=beta
        self.args.lammbda=lammbda
        self.model = DMG(args)
        self.results = None

    def fit(self,X_train, y_train):
        # 直接调用原始模型获取最终结果的方法
        self.results = self.model.training()
        return self

    def predict(self, X):
        # 这里假设可以根据输入X从已保存的结果中获取对应的预测值
        # 具体实现需要根据原始模型的结果结构和预测逻辑来确定
        adata=clustering(self.results, self.args.dataset, self.args.alpha,self.args.beta, self.args.lammbda, self.args.num_iters, self.args.input_folder,n_clusters=self.args.n_clusters,use_pca=True,model=self.args.model)
        predictions = adata.obs["cluster_result"]
        return predictions

    def get_params(self,deep=True):
        return self.model.get_params(deep) if hasattr(self.model, 'get_params') else {}

    def set_params(self, alpha,beta,lammbda):
        if hasattr(self.model, 'set_params'):
            self.model.set_params(alpha,beta,lammbda)
        else:
            for key, value in params.items():
                setattr(self.model, key, value)
        return self

    # def set_params(self, params):
    #     if hasattr(self.model, 'set_params'):
    #         self.model.set_params(params)
    #     else:
    #         for key, value in params.items():
    #             setattr(self.model, key, value)
    #     return self

# 定义自己的评估指标，这里以平均绝对误差为例
def custom_mae(y_true, y_pred):
    print(y_pred)
    dataset="human_lymph_node_A1"
    input_folder = "/home/lixiangyu/DMG/DMG-main/input/"+dataset+'/'
    model = "resnet_v2"
    distance = "cos"
    print(dataset)
    f1 = input_folder + dataset + "_Expression_HVG3000.csv"
    truefeatures_1 = pd.read_csv(f1, sep=",", na_filter=False, index_col=0)

    if dataset=='human_lymph_node_A1':
        # 读取第一个文件的数据
        f1 = input_folder + dataset + "_ADT.csv"
        truefeatures_2 = pd.read_csv(f1, sep=",", na_filter=False, index_col=0)
    else:
        truefeatures_2 = pd.read_csv(input_folder + dataset + "_GeneScoreMatrix_HVG3000.csv", sep=",", na_filter=False, index_col=0)
    
    # 读取第三个文件的数据
    f3 = input_folder + dataset + "_image_" + model + "_" + distance + "_normalize.csv"
    truefeatures_3 = pd.read_csv(f3, sep=",", na_filter=False, index_col=0)
    truefeatures_1=truefeatures_1.values
    truefeatures_2=truefeatures_2.values
    truefeatures_3=truefeatures_3.values
    #print(truefeatures_1,truefeatures_2,truefeatures_3)
    # 合并三个数据集（这里假设它们可以按行或按列进行合并，根据实际情况调整axis参数）
    # 如果是按行合并，示例如下，若按列合并则axis = 1
    combined_features = np.concatenate((truefeatures_1, truefeatures_2,truefeatures_3), axis = 1)
    print(combined_features.shape)
    X = combined_features
    mask = np.isnan(X)
    X[mask] = 0
    # 数据标准化（可选，但对于很多指标计算在高维等情况下有帮助）
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    ch_index = calinski_harabasz_score(X, y_pred)
    print("ch_index",ch_index)
  
    return ch_index

    
if __name__ == '__main__':
    datasets_name = ["E11_0", "E13_5", "E15_5", "E18_5", "mouse_spleen","mouse_embryos","mouse_brain","human_lymph_node_A1"]
    args = get_args()
    printConfig(args)
    # embedder = DMG(args)
    # embeddings = embedder.training()
    # 创建一个可用于RandomizedSearchCV的评估器对象
    scorer = make_scorer(custom_mae, greater_is_better=True)
    
    # 定义超参数分布
    param_distributions = {
        'alpha': [0.5,0.6,0.7,0.8,0.9,1.0],
        'beta': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
        'lammbda':[0.5,0.6,0.7,0.8,0.9,1.0]
    }
    # 创建WrappedModel对象作为估计器
    wrapped_model = WrappedModel(args,args.alpha,args.beta,args.lammbda)
    
    # 创建RandomizedSearchCV对象，并指定评估指标
    random_search = RandomizedSearchCV(estimator=wrapped_model, param_distributions=param_distributions, n_iter=1, cv=2, scoring=scorer)
    # input_folder
    # dataset="human_lymph_node_A1"
    data = pd.read_csv(args.input_folder+args.dataset+"_groundtruth.csv")
    
    input_folder = "/hy-tmp/DMG-main/input/"+args.dataset+'/'
    model = "resnet_v2"
    distance = "cos"
    f1 = args.input_folder + args.dataset + "_Expression_HVG3000.csv"
    truefeatures_1 = pd.read_csv(f1, sep=",", na_filter=False, index_col=0)
    X_train=truefeatures_1
    y_train=data['Ground_Truth'].values
    random_search.fit(X_train, y_train)
    print("Best parameters: ", random_search.best_params_)
    print("Best score: ", random_search.best_score_)
