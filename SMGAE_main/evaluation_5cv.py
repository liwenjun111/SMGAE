from sklearn.model_selection import KFold
import random
import copy
from tqdm import tqdm
import torch
torch.set_printoptions(sci_mode=False)
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn import metrics
from SMGAE_main.utils import create_optimizer, accuracy, f1_score, AUC, AUPRC
import numpy as np
from GAE_model.train import embedding

def node_classification_evaluation_5cv(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device,linear_prob=True, mute=False):
    model.eval()
    if linear_prob:
        with torch.no_grad():
            in_feat =embedding.shape[1]
            print(f"feat_size:", in_feat)
        encoder = LogisticRegression(in_feat)
    else:
        embedded_x = model.embed(graph.to(device), x.to(device))
        encoder = model.encoder
        encoder.reset_classifier(num_classes)

    num_finetune_params = [p.numel() for p in encoder.parameters() if p.requires_grad]
    if not mute:
        print(f"num parameters for finetuning: {sum(num_finetune_params)}")
    encoder.to(device)
    optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
    final_auc, final_auprc = linear_probing_for_transductive_node_classiifcation(encoder, graph, embedding, optimizer_f, max_epoch_f, device, mute)
    return final_auc, final_auprc

def linear_probing_for_transductive_node_classiifcation(model, graph, feat, optimizer, max_epoch, device, mute=False):

    graph = graph.to(device)
    #x = feat.to(device)
    # 从CSV文件加载标签数据
    labels_df = pd.read_csv("/data/datasets/label.csv")
    labels = labels_df["label"].values
    labels = torch.tensor(labels)
    labels = labels.to(device)
    # 将标签数据设置为Graph的节点标签
    graph.ndata["label"] = labels
    labels =  graph.ndata["label"]
    labeled_nodes_mask = (labels != -1)  # 只评估已标记节点
    labeled_labels = labels[labeled_nodes_mask]

    # 随机欠采样
    #################################################
    # 统计正样本和负样本数量
    positive_count = (labels == 1).sum().item()
    negative_count = (labels == 0).sum().item()
    # 获取负样本索引
    negative_indices = torch.where(labels == 0)[0]
    # 随机修改大部分负样本的标签为-1
    selected_negative_indices = random.sample(negative_indices.tolist(), negative_count - positive_count)
    #selected_negative_indices = random.sample(negative_indices.tolist(), 1500)
    labels[selected_negative_indices] = -1
    labeled_nodes_mask = (labels != -1)
    labeled_labels = labels[labeled_nodes_mask]
    ##################################################

    # 统计正样本和负样本数量
    pos_num = (labeled_labels == 1).sum().item()
    neg_num = (labeled_labels == 0).sum().item()
    print(pos_num, neg_num)
    weight = torch.tensor([neg_num / pos_num])
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight).to(device)


    subgraph = graph.subgraph(labeled_labels)
    nodes = subgraph.nodes()
    # 定义K折交叉验证的折数
    n_splits = 5

    # 创建K折交叉验证迭代器
    kf = KFold(n_splits=n_splits, shuffle=True)

    # 存储每个折的测试结果

    train_aucs = []
    test_aucs = []
    train_auprcs = []
    test_auprcs = []
    best_test_auprcs = []
    best_test_aucs = []
    estp_test_aucs = []
    estp_test_auprcs = []

    # 划分已标记节点的数据集
    train_mask, test_mask = train_val_test_split(graph.subgraph(labeled_nodes_mask), test_size=0.2)
    # 将 NumPy 数组转换为 DGL 张量
    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)
    train_mask = train_mask.to(device)
    test_mask = test_mask.to(device)
    subgraph = graph.subgraph(labeled_nodes_mask)
    subgraph.ndata["train_mask"], subgraph.ndata["test_mask"] = train_mask, test_mask
    subgraph = subgraph.to(device)
    train_mask = subgraph.ndata["train_mask"]
    test_mask = subgraph.ndata["test_mask"]

    # 进行K折交叉验证
    for fold, (train_index, test_index) in enumerate(kf.split(nodes, labeled_labels)):
     for run in range(1, 10 + 1):


        # 获取当前折的训练集和测试集
        train_mask = np.isin(range(len(nodes)), train_index)
        test_mask = np.isin(range(len(nodes)), test_index)
        # 将 NumPy 数组转换为 DGL 张量
        train_mask = torch.tensor(train_mask, dtype=torch.bool)
        test_mask = torch.tensor(test_mask, dtype=torch.bool)
        train_mask = train_mask.to(device)
        test_mask = test_mask.to(device)
        subgraph.ndata["train_mask"], subgraph.ndata["test_mask"] = train_mask, test_mask
        train_mask = subgraph.ndata["train_mask"]
        test_mask = subgraph.ndata["test_mask"]



        x = feat.to(device)
        x_label = x[labeled_nodes_mask]
        x_label_train = x_label[train_mask]
        label_train = labeled_labels[train_mask]
        x_label = x[labeled_nodes_mask]
        x_label_test = x_label[test_mask]
        label_test = labeled_labels[test_mask]

        best_test_auprc = 0
        best_test_epoch = 0
        best_model = None

        if not mute:
            epoch_iter = tqdm(range(max_epoch))
        else:
            epoch_iter = range(max_epoch)

        for epoch in epoch_iter:
            model.train()
            #out = model(x)
            #x_label = x[labeled_nodes_mask]
            #x_label_train = x_label[train_mask]
            out = model(x_label_train)
            #out = out[:, 1]
            #label_train = label_train.float()
            label_train = label_train.float()
            loss = criterion(out, label_train)
            #loss = F.binary_cross_entropy_with_logits(out, label_train)
            #train_acc = accuracy(labeled_out[train_mask], labeled_labels[train_mask])

            train_auc, train_auprc, train_precision, train_recall = result(out.cpu(), label_train.cpu())
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            with torch.no_grad():
                model.eval()
                #pred = model(x)
                #x_label = x[labeled_nodes_mask]
                #x_label_test = x_label[test_mask]
                pred = model(x_label_test)
                #pred = pred[:, 1]
                label_test = label_test.float()
                test_loss = criterion(pred, label_test)
                #test_loss = F.binary_cross_entropy_with_logits(pred, label_test)
                test_auc, test_auprc, precision, recall = result(pred.cpu(), label_test.cpu())

            if test_auprc >= best_test_auprc:
                best_test_auc = test_auc
                best_test_auprc = test_auprc
                best_test_epoch = epoch
                best_model = copy.deepcopy(model)

            if not mute:
                epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f},  test_loss:{test_loss.item(): .4f}")

        best_model.eval()
        with torch.no_grad():
              #pred = best_model(x)
              x_label = x[labeled_nodes_mask]
              x_label_test = x_label[test_mask]
              pred = best_model(x_label_test)
              #print(pred)

              label_test = label_test.float()
              #print(pred)
              estp_test_auc, estp_test_auprc, precision_f, recall_f =result(pred.cpu(), label_test.cpu())


        # 记录当前折的测试准确率
        train_aucs.append(train_auc)
        test_aucs.append(test_auc)
        train_auprcs.append(train_auprc)
        test_auprcs.append(test_auprc)
        best_test_aucs.append(best_test_auc)
        best_test_auprcs.append(best_test_auprc)
        estp_test_aucs.append(estp_test_auc)
        estp_test_auprcs.append(estp_test_auprc)

    if mute:
        print(f"# IGNORE: --- Pretrain End--- ")
    else:
        # 打印每个折的测试结果
        for run in range(10):
          print( f"--- Run {run + 1}: TrainAuc: {train_aucs[run]:.4f}, TestAuc: {test_aucs[run]:.4f}, Best testAUC: {best_test_aucs[run]:.4f} ,estp testAUC: {estp_test_aucs[run]:.4f} ,TrainAuprc: {train_auprcs[run]:.4f}, TestAuprc: {test_auprcs[run]:.4f},Best testAUPRC: {best_test_auprcs[run]:.4f} ,estp testAUPRC: {estp_test_auprcs[run]:.4f} , in epoch {best_test_epoch} ---")

    # 计算平均准确率

    avg_test_auc, avg_test_auc_std = np.mean(np.squeeze(np.array(test_aucs))), np.std(np.squeeze(np.array(test_aucs)))
    avg_test_auprc, avg_test_auprc_std = np.mean(np.squeeze(np.array(test_auprcs))), np.std(np.squeeze(np.array(test_auprcs)))

    print(f"--- Average TestAuc: {avg_test_auc:.4f}±{avg_test_auc_std:.4f}, Average TestAuprc: {avg_test_auprc:.4f}±{avg_test_auprc_std:.4f}---")

    return  avg_test_auc, avg_test_auprc


class LogisticRegression(nn.Module):
    def __init__(self, num_dim):
        super().__init__()
        self.linear = nn.Linear(num_dim, 1)

    def forward(self, x):
        logits = self.linear(x)

        return logits.squeeze()

def result(pred, true):
    aa = torch.sigmoid(pred)
    precision, recall, _thresholds = metrics.precision_recall_curve(true, aa.detach().numpy())
    area = metrics.auc(recall, precision)
    return metrics.roc_auc_score(true, aa.detach().numpy()), area, precision, recall


def train_val_test_split(graph, test_size=0.2):
    num_nodes = graph.num_nodes()
    perm = np.random.permutation(num_nodes)
    test_size = int(test_size * num_nodes)
    test_mask = np.zeros(num_nodes, dtype=bool)
    test_mask[perm[:test_size]] = True
    train_mask = np.logical_not(test_mask)

    return train_mask, test_mask