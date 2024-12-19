
from collections import namedtuple, Counter
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

import dgl
from sklearn.preprocessing import StandardScaler



def preprocess(graph):
    feat = graph.ndata["feat"]
    # 去除自环
    graph = dgl.remove_self_loop(graph)
    # 去除重复边并转换为简单图
    graph = dgl.to_simple(graph)
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()
    return graph


def scale_feats(x):

    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats


def load_dataset():

        # 从CSV文件加载边列表
        edges_df = pd.read_csv("/data/datasets/CPDB.csv")
        src_nodes = edges_df.iloc[:, 0].values
        dst_nodes = edges_df.iloc[:, 1].values
        # 创建DGL Graph对象
        graph = dgl.graph((src_nodes, dst_nodes))


        # 从CSV文件加载生物特征数据
        features_df = pd.read_csv("/data/datasets/feature.csv")
        feat_bio = features_df.values
        feat_bio = torch.tensor(feat_bio)
        feat_bio = scale_feats(feat_bio)

        # 将特征数据设置为Graph的节点特征
        graph.ndata["feat"] = feat_bio

        # 图预处理
        graph = preprocess(graph)
        # 特征标准化
        feat = graph.ndata["feat"]
        feat = scale_feats(feat)
        graph.ndata["feat"] = feat

        num_features = graph.ndata["feat"].shape[1]
        print("特征维度:",num_features)
        num_classes = 2
        return graph, (num_features, num_classes)



