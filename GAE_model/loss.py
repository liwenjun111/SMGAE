import torch
import torch.nn.functional as F

def auc_loss(pos_out, neg_out):
    return torch.square(1 - (pos_out - neg_out)).sum()

def hinge_auc_loss(pos_out, neg_out):
    return (torch.square(torch.clamp(1 - (pos_out - neg_out), min=0))).sum()

def log_rank_loss(pos_out, neg_out, num_neg=1):
    return -torch.log(torch.sigmoid(pos_out - neg_out) + 1e-15).mean()

def ce_loss(pos_out, neg_out):
    pos_loss = F.binary_cross_entropy(pos_out.sigmoid(), torch.ones_like(pos_out))
    neg_loss = F.binary_cross_entropy(neg_out.sigmoid(), torch.zeros_like(neg_out))
    return pos_loss + neg_loss

def info_nce_loss(pos_out, neg_out):
    pos_exp = torch.exp(pos_out)
    neg_exp = torch.sum(torch.exp(neg_out), 1, keepdim=True)
    return -torch.log(pos_exp / (pos_exp + neg_exp) + 1e-15).mean()

def sce_loss(x, y, alpha=1):
    #L2归一化
    #x = F.normalize(x, p=2, dim=-1)
    #y = F.normalize(y, p=2, dim=-1)
    #loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    #L1归一化
    #x = F.normalize(x, p=1, dim=-1)
    #y = F.normalize(y, p=1, dim=-1)
    #计算x,y模长
    x_norm = torch.norm(x, dim=-1)
    y_norm = torch.norm(y, dim=-1)
    cos_sim = (x * y).sum(dim=-1) / (x_norm * y_norm)
    loss = (1 - cos_sim).pow_(alpha)


    loss = loss.mean()
    return loss

def MyMSEloss(x, y):
    #x = F.normalize(x, p=2, dim=-1)
    #y = F.normalize(y, p=2, dim=-1)
    #MSE
    loss = ((x - y) ** 2)
    loss = loss.mean()
    return loss
