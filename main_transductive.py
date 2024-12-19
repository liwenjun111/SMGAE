import logging
import numpy as np
from tqdm import tqdm
import torch

from SMGAE_main.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)

from SMGAE_main.datasets.data_util import load_dataset
from SMGAE_main.evaluation_5cv import node_classification_evaluation_5cv
from SMGAE_main.models import build_model


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)




def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate

    optim_type = args.optimizer 
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler

    graph, (num_features, num_classes) = load_GAEdataset()

    auc_list = []
    auprc_list = []

    for i, seed in enumerate(seeds):
            print(f"####### Run {i} for seed {seed}")
            set_random_seed(seed)

            if logs:
                logger = TBLogger(
                    name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
            else:
                logger = None

            model = build_model(args)
            model.to(device)
            optimizer = create_optimizer(optim_type, model, lr, weight_decay)

            if use_scheduler:
                logging.info("Use schedular")
                scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5
                scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
            else:
                scheduler = None

            x = graph.ndata["feat"]
            if not load_model:
                model = model.cpu()

            if load_model:
                logging.info("Loading Model ... ")
                model.load_state_dict(torch.load("checkpoint.pt"))
            if save_model:
                logging.info("Saveing Model ...")
                torch.save(model.state_dict(), "checkpoint.pt")

            model = model.to(device)
            model.eval()
            final_auc, final_auprc = node_classification_evaluation_5cv(model, graph, x, num_classes, lr_f,weight_decay_f, max_epoch_f, device,linear_prob)
            auc_list.append(final_auc)
            auprc_list.append(final_auprc)
            if logger is not None:
                logger.finish()


    final_auc, final_auc_std = np.mean(auc_list), np.std(auc_list)
    final_auprc, final_auprc_std = np.mean(auprc_list), np.std(auprc_list)
    print(f"#final_auc: {final_auc:.4f}±{final_auc_std:.4f}")
    print(f"#final_auprc: {final_auprc:.4f}±{final_auprc_std:.4f}")

    return final_auc, final_auprc

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    main(args)


