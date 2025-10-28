from __future__ import print_function
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--exp_array', default='../Data/human_lymph_node_A1/RNA_exp_array.csv')
parser.add_argument('--pos_adj', default='../Data/human_lymph_node_A1/adj_human_lymph_node_A1_position.csv')
parser.add_argument('--pos_dis', default='../Data/human_lymph_node_A1/cor_human_lymph_node_A1_position.csv')
parser.add_argument('--image_adj', default='../Data/human_lymph_node_A1/adj_human_lymph_node_A1_resnet50_cos.csv')
parser.add_argument('--image_dis', default='../Data/human_lymph_node_A1/cor_human_lymph_node_A1_resnet50_cos.csv')
parser.add_argument('--destination', default='../Data/human_lymph_node_A1/RNA_human_lymph_node_A1_resnet50_cos.csv')
args = parser.parse_args()

print(args.destination)

# 读取第一个邻接矩阵
adj1 = pd.read_csv(args.pos_adj, header=None)
adj1 = np.array(adj1)
adj1 = np.delete(adj1, 0, axis=0)
adj1 = np.delete(adj1, 0, axis=1)
adj1 = adj1.astype(np.float64)

# 读取第二个邻接矩阵
adj2 = pd.read_csv(args.image_adj, header=None)
adj2 = np.array(adj2)
adj2 = np.delete(adj2, 0, axis=0)
adj2 = np.delete(adj2, 0, axis=1)
adj2 = adj2.astype(np.float64)

# 获取两个邻接矩阵的行数和列数
adj1_row, adj1_column = adj1.shape
adj2_row, adj2_column = adj2.shape

print(adj1_row, adj1_column)
print("Reading pos_adj completed.")

print(adj2_row, adj2_column)
print("Reading image_adj completed.")

# 创建一个新的邻接矩阵来存储相同的边
common_adj = np.zeros_like(adj1)

# 遍历两个邻接矩阵，找出相同的边
for i in range(adj1_row):
    for j in range(adj1_column):
        if adj1[i][j]!= 0 and adj2[i][j]!= 0:
            common_adj[i][j] = adj1[i][j]

# 现在 common_adj 就是只包含相同边的邻接矩阵
# 将 common_adj 保存为 CSV 文件
pd.DataFrame(common_adj).to_csv(args.destination, header=None, index=None)
print(f"Saved common adjacency matrix to {args.destination}")
