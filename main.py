from models import PepBAN
from time import time
from utils import set_seed, mkdir
from configs import get_cfg_defaults
from dataloader import PepPIDataset, MultiDataLoader
from torch.utils.data import DataLoader
from trainer import Trainer
from domain_adaptator import Discriminator, DANN
import torch
import argparse
import warnings, os
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="PepBAN for PepPI prediction")
# parser.add_argument('--cfg', help="path to config file", type=str, default='configs/STRING.yaml')
parser.add_argument('--seed', type=int, default=32)
parser.add_argument('--data', required=True, type=str, help='dataset')
args = parser.parse_args()


def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    # cfg.merge_from_file(args.cfg)
    cfg.merge_from_file(f"configs/camp.yaml")
    set_seed(args.seed)
    suffix = str(int(time() * 1000))[6:]
    mkdir(cfg.RESULT.OUTPUT_DIR)
    # print(f"Config yaml: {args.cfg}")
    # print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    dataFolder = f'datasets/{args.data}'

    train_source_path = os.path.join(dataFolder, 'source_train.csv')
    train_target_path = os.path.join(dataFolder, 'target_train.csv')
    test_target_path = os.path.join(dataFolder, 'target_test.csv')
    
    df_train_source = pd.read_csv(train_source_path)
    df_train_target = pd.read_csv(train_target_path)
    df_test_target = pd.read_csv(test_target_path)

    train_dataset = PepPIDataset(df_train_source.index.values, df_train_source, args.data)
    train_target_dataset = PepPIDataset(df_train_target.index.values, df_train_target, args.data)
    test_target_dataset = PepPIDataset(df_test_target.index.values, df_test_target, args.data)



    hyper_params = {
        "LR": cfg.SOLVER.LR,
        "Output_dir": cfg.RESULT.OUTPUT_DIR,
        "DA_use": cfg.DA.USE,
        "DA_task": cfg.DA.TASK,
    }
    if cfg.DA.USE:
        da_hyper_params = {
            "DA_init_epoch": cfg.DA.INIT_EPOCH,
            "Use_DA_entropy": cfg.DA.USE_ENTROPY,
            "DA_optim_lr": cfg.SOLVER.DA_LR
        }
        hyper_params.update(da_hyper_params)

    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'drop_last': True}


    if not cfg.DA.USE:
        training_generator = DataLoader(train_dataset, **params)
        params['shuffle'] = False
        params['drop_last'] = False
        test_generator = DataLoader(test_target_dataset, **params)
    else:
        source_train_generator = DataLoader(train_dataset, **params)
        target_train_generator = DataLoader(train_target_dataset, **params)
        n_batches = max(len(source_train_generator), len(target_train_generator))
        multi_generator = MultiDataLoader(dataloaders=[source_train_generator, target_train_generator], n_batches=n_batches)
        params['shuffle'] = False
        params['drop_last'] = False
        test_generator = DataLoader(test_target_dataset, **params)

    model = PepBAN(**cfg).to(device)

    if cfg.DA.USE:
        domain_dmm = Discriminator(input_size=cfg["DA"]["RANDOM_DIM"], n_class=cfg["DECODER"]["BINARY"]).to(device)
        # params = list(model.parameters()) + list(domain_dmm.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
        opt_da = torch.optim.Adam(domain_dmm.parameters(), lr=cfg.SOLVER.DA_LR)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)

    torch.backends.cudnn.benchmark = True

    if not cfg.DA.USE:
        trainer = Trainer(args.data, model, opt, device, training_generator, test_generator, opt_da=None,
                          discriminator=None, **cfg)
    else:
        trainer = Trainer(args.data, model, opt, device, multi_generator, test_generator, opt_da=opt_da,
                          discriminator=domain_dmm, **cfg)
    result = trainer.train()

    return result


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")