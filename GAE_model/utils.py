import sys
import os
import pandas as pd
import torch
import random
import numpy as np
from texttable import Texttable
import torch_geometric.transforms as T
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, Reddit
from torch_geometric.data import Data
from torch_geometric.utils import index_to_mask
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False




def get_PPIdataset(root: str, node_feature_file: str, edge_index_file: str, target_label_file: str) -> Data:
    # Load node features from CSV file
    node_features_df = pd.read_csv(os.path.join(root, node_feature_file))
    x = torch.tensor(node_features_df.values, dtype=torch.float)
    x = scale_feats(x)

    # Load edge indices from CSV file
    edge_index_df = pd.read_csv(os.path.join(root, edge_index_file))
    edge_index = torch.tensor(edge_index_df.iloc[:, :2].values.T, dtype=torch.long)

    # Load target labels from CSV file
    target_labels_df = pd.read_csv(os.path.join(root, target_label_file))
    y = torch.tensor(target_labels_df['label'].values, dtype=torch.long)

    num_nodes = len(node_features_df)

    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

    # Generate index train, validation, and test index for node classification
    # 过滤出目标标签不为-1的数据
    filtered_data = target_labels_df[target_labels_df['label'] != -1]
    # 获取过滤后的索引
    filtered_indices = filtered_data.index
    # 通过索引过滤特征数据
    filtered_node_features_df = node_features_df.loc[filtered_indices]

    labeled_x = torch.tensor(filtered_node_features_df.values, dtype=torch.float)
    labeled_y = torch.tensor(filtered_data['label'].values, dtype=torch.long)
    labeled_num_nodes = len(filtered_node_features_df)
    clf_data = Data(labeled_x=labeled_x, labeled_y=labeled_y, labeled_num_nodes=labeled_num_nodes, filtered_indices=filtered_indices)

    train_idx, test_idx = train_test_split(range(clf_data.labeled_num_nodes), test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)
    train_idx = torch.tensor(train_idx)
    val_idx = torch.tensor(val_idx)
    test_idx = torch.tensor(test_idx)
    clf_data.train_mask = index_to_mask(train_idx, clf_data.labeled_num_nodes)
    clf_data.val_mask = index_to_mask(val_idx, clf_data.labeled_num_nodes)
    clf_data.test_mask = index_to_mask(test_idx, clf_data.labeled_num_nodes)

    return data, clf_data




def tab_printer(args):
    """Function to print the logs in a nice tabular format.

    Note
    ----
    Package `Texttable` is required.
    Run `pip install Texttable` if was not installed.

    Parameters
    ----------
    args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k, str(args[k])] for k in keys if not k.startswith('__')])
    return t.draw()


def scale_feats(x):

    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats