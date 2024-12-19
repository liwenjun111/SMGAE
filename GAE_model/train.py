import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from tqdm.auto import tqdm

from torch.utils.data import DataLoader
# custom modules
from GAE_model.utils import set_seed, tab_printer, get_PPIdataset
from GAE_model.model import MaskGAE, NodeDecoder, EdgeDecoder, GNN
from GAE_model.mask import MaskEdge, MaskPath, MaskNode
from SMGAE_main.utils import build_args

def train_linkpred(model, splits, args, device="cpu"):
    print('Start Training (Link Prediction Pretext Training)...')
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr_e,
                                 weight_decay=args.weight_decay_e)

    best_valid = 0
    batch_size = args.batch_size
    
    train_data = splits['train'].to(device)
    valid_data = splits['valid'].to(device)
    test_data = splits['test'].to(device)
    
    model.reset_parameters()
    
    for epoch in tqdm(range(1, 1 + args.epochs)):

        loss = model.train_step(train_data, optimizer,
                                alpha=args.alpha, 
                                batch_size=args.batch_size)
        
        if epoch % args.eval_period == 0:
            valid_auc, valid_ap = model.test_step(valid_data, 
                                                  valid_data.pos_edge_label_index, 
                                                  valid_data.neg_edge_label_index, 
                                                  batch_size=batch_size)
            if valid_auc > best_valid:
                best_valid = valid_auc
                best_epoch = epoch
                torch.save(model.state_dict(), args.save_path)

    #model.load_state_dict(torch.load(args.save_path))
    test_auc, test_ap = model.test_step(test_data,
                                        test_data.pos_edge_label_index,
                                        test_data.neg_edge_label_index,
                                        batch_size=batch_size)
    
    print(f'Link Prediction Pretraining Results:\n'
          f'AUC: {test_auc:.2%}',
          f'AP: {test_ap:.2%}')
    return test_auc, test_ap




args = build_args()
print(args)

set_seed(args.seed_e)

if args.device < 0:
    device = "cpu"
else:
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

transform = T.Compose([
    T.ToUndirected(),
    T.ToDevice(device),
])


########## (!IMPORTANT) Specify the path to your dataset directory ##########
# root = '/data/datasets' # my root directory
#############################################################################

data, clf_data = get_PPIdataset('/data/datasets', 'feature.csv', 'CPDB.csv', 'label.csv')

print(data)
print(clf_data)
train_data, val_data, test_data = T.RandomLinkSplit(num_val=0.1, num_test=0.05,
                                                    is_undirected=True,
                                                    split_labels=True,
                                                    add_negative_train_samples=False)(data)


# Use full graph for pretraining
splits = dict(train=data, valid=val_data, test=test_data)

if args.mask == 'Path':
    mask1 = MaskPath(p=args.p,
                    num_nodes=data.num_nodes, 
                    start=args.start,
                    walk_length=args.encoder_layers+1)
elif args.mask == 'Edge':
    mask1 = MaskEdge(p=args.p)
else:
    mask1 = None  # vanilla GAE

mask2 = MaskNode(q=args.q)

encoder = GNN(data.num_features, args.encoder_channels, args.hidden_channels,
                     num_layers=args.encoder_layers, dropout=args.encoder_dropout,
                     bn=args.bn, layer=args.layer, activation=args.encoder_activation)

edge_decoder = EdgeDecoder(args.hidden_channels, args.decoder_channels,
                           num_layers=args.decoder_layers, dropout=args.decoder_dropout)


node_decoder = NodeDecoder(args.hidden_channels, args.decoder_channels,
                               num_layers=args.decoder_layers, dropout=args.decoder_dropout)

model = MaskGAE(encoder, edge_decoder, node_decoder, mask1, mask2).to(device)

train_linkpred(model, splits, args, device=device)

data = data.to(device)
embedding = model.encoder.get_embedding(data.x, data.edge_index)
if args.l2_normalize:
    embedding = F.normalize(embedding_edge, p=2, dim=1)
