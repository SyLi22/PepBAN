from itertools import product
from models import PepBAN
from time import time
from utils import set_seed, mkdir
from configs import get_cfg_defaults
from dataloader import PepPIDataset, MultiDataLoader
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from trainer import Trainer
from domain_adaptator import Discriminator
import torch
import argparse
import warnings, os
import pandas as pd
import random
import numpy as np
import csv
from tqdm import tqdm
import yaml
from sklearn.model_selection import train_test_split
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="PepBAN for PepPI prediction")
parser.add_argument('--data', type=str)
parser.add_argument('--seed', type=int, default=32)
args = parser.parse_args()
torch.cuda.set_device(0)

def train_single_config(cfg, dataset_name, config_id):
    print(f"\nTraining config {config_id} on dataset {dataset_name}: {dict(cfg)}")
    
    dataFolder = f'datasets/{dataset_name}'
    train_source_path = os.path.join(dataFolder, 'source_train.csv')
    train_target_path = os.path.join(dataFolder, 'target_train.csv')
    # test_target_path = os.path.join(dataFolder, 'target_test.csv')
    
    df_train_source = pd.read_csv(train_source_path)
    df_train_target = pd.read_csv(train_target_path)
    # df_test_target = pd.read_csv(test_target_path)
    df_train_target, df_test_target=train_test_split(df_train_target, test_size=0.2, random_state=42)
    df_train_target=df_train_target.reset_index(drop=True)
    df_test_target=df_test_target.reset_index(drop=True)
    train_dataset = PepPIDataset(df_train_source.index.values, df_train_source, dataset_name)
    train_target_dataset = PepPIDataset(df_train_target.index.values, df_train_target, dataset_name)
    test_target_dataset = PepPIDataset(df_test_target.index.values, df_test_target, dataset_name)

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
        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
        opt_da = torch.optim.Adam(domain_dmm.parameters(), lr=cfg.SOLVER.DA_LR)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)

    torch.backends.cudnn.benchmark = True

    if not cfg.DA.USE:
        trainer = Trainer(dataset_name, model, opt, device, training_generator, test_generator, 
                         opt_da=None, discriminator=None, **cfg)
    else:
        trainer = Trainer(dataset_name, model, opt, device, multi_generator, test_generator, 
                         opt_da=opt_da, discriminator=domain_dmm, **cfg)
    
    # metrics = trainer.train()
    auroc, auprc, sensitivity, specificity, accuracy, _, precision, mcc, f1, recall = trainer.train()
    
    # 返回字典
    return {
        "auroc": auroc,
        "auprc": auprc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "precision": precision,
        "mcc": mcc,
        "f1": f1,
        "recall": recall
    }

def random_search(cfg, datasets, param_grid, results_file, num_samples=10, num_runs=1):
    set_seed(args.seed)
    results = []
    keys = list(param_grid.keys())
    
    for _ in tqdm(range(num_samples)):
        sampled_values = {}
        combo = []
        
        for key in keys:
            values = param_grid[key]
            
            if isinstance(values, list):
                value = random.choice(values)
            elif isinstance(values, tuple) and len(values) == 2:
                low, high = values
                if any(isinstance(x, float) for x in values):
                    value = random.uniform(low, high)
                else:
                    value = random.randint(low, high)
            else:
                raise ValueError(f"Unsupported parameter format: {key}: {values}")
            
            combo.append(value)
            sampled_values[key] = value
        
        temp_cfg = cfg.clone()
        for key, value in sampled_values.items():
            keys_hierarchy = key.split('.')
            node = temp_cfg
            for k in keys_hierarchy[:-1]:
                node = node[k]
            node[keys_hierarchy[-1]] = value
            
            cumulative_metrics = {
                "auroc": 0, "auprc": 0, "accuracy": 0, "precision": 0, 
                "recall": 0, "f1": 0, "specificity": 0, "sensitivity": 0, "mcc": 0
            }

        for run in range(num_runs):
            metrics = train_single_config(temp_cfg, datasets, f"sample_{_}_run_{run}_dataset_{datasets}")
            for metric in cumulative_metrics:
                cumulative_metrics[metric] += metrics[metric]
        
        avg_metrics = {metric: value / num_runs for metric, value in cumulative_metrics.items()}
        
        result_row = {
            "params": str(sampled_values),
            **avg_metrics
        }
        
        results.append(result_row)
        save_results(result_row, results_file)
    
    return pd.DataFrame(results)

def save_results(result_row, filepath='result/all_datasets_results.csv'):
    file_exists = os.path.exists(filepath)
    with open(filepath, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=result_row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_row)

def analyze_results(results_file, original_cfg, yaml_file):
    df = pd.read_csv(results_file)

    
    print("\n=== Best Overall Parameters ===")
    best_overall = df.loc[df['auroc'].idxmax()]
    # print(best_overall[['dataset', 'params', 'auroc', 'auprc']])
    

    # avg_performance = df.groupby('params').mean().reset_index()
    # print("\n=== Parameters with Best Average Performance ===")
    # best_avg = avg_performance.loc[avg_performance['auroc'].idxmax()]
    # print(best_avg[['params', 'auroc', 'auprc']])
    import ast
    from collections import OrderedDict
    best_params = ast.literal_eval(best_overall['params'])
    # print(best_params)
    best_cfg = original_cfg.clone()
    
    for key, value in best_params.items():
        keys = key.split('.')
        node = best_cfg
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value
    
    yaml_content = OrderedDict([
        ('SOLVER', OrderedDict([
            ('BATCH_SIZE', best_cfg.SOLVER.BATCH_SIZE),
            ('LR', float(f"{best_cfg.SOLVER.LR:.8f}")),
            ('DA_LR', float(f"{best_cfg.SOLVER.DA_LR:.8f}")),
        ])),
        ('DA', OrderedDict([
            ('RANDOM_DIM', best_cfg.DA.RANDOM_DIM),
            ('INIT_EPOCH', best_cfg.DA.INIT_EPOCH)
        ])),
        ('DECODER', OrderedDict([
            ('IN_DIM', best_cfg.DECODER.IN_DIM),
            ('HIDDEN_DIM', best_cfg.DECODER.HIDDEN_DIM),
            ('OUT_DIM', best_cfg.DECODER.OUT_DIM),
        ])),
        ('PEPTIDE', OrderedDict([
            ('EMBEDDING_DIM', best_cfg.PEPTIDE.EMBEDDING_DIM),
            ('NUM_FILTERS', best_cfg.PEPTIDE.NUM_FILTERS),
        ])),
        ('PROTEIN', OrderedDict([
            ('EMBEDDING_DIM', best_cfg.PROTEIN.EMBEDDING_DIM),
            ('NUM_FILTERS', best_cfg.PROTEIN.NUM_FILTERS),
        ])),
        ('BCN', OrderedDict([
            ('HEADS', best_cfg.BCN.HEADS),
        ]))
    ])
    

    def setup_yaml():
        yaml.add_representer(
            OrderedDict,
            lambda dumper, data: dumper.represent_mapping(
                'tag:yaml.org,2002:map', data.items()))
        
        def float_representer(dumper, value):
            text = "{:.8f}".format(value).rstrip('0').rstrip('.')
            return dumper.represent_scalar('tag:yaml.org,2002:float', text)
        
        yaml.add_representer(float, float_representer)
    
    setup_yaml()
    
    # 写入YAML文件
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False, width=1000)

    # print(f"\n最佳参数已保存到: {output_yaml}")
    return best_cfg



if __name__ == '__main__':
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    # cfg.merge_from_file(args.cfg)
    set_seed(args.seed)

    
    # parameters space
    param_grid = {
        "SOLVER.BATCH_SIZE": [32, 64, 128, 256],
        "SOLVER.LR": (1e-5, 1e-3),
        "SOLVER.DA_LR": (1e-5, 1e-3),
        "DA.RANDOM_DIM": [128, 256, 512],
        "DA.INIT_EPOCH": (5, 20),
        "DECODER.IN_DIM": [128, 256, 512],
        "DECODER.HIDDEN_DIM": [128, 256, 512],
        "DECODER.OUT_DIM": [128, 256, 512],
        "PEPTIDE.EMBEDDING_DIM": [128, 256, 512],
        "PROTEIN.EMBEDDING_DIM": [128, 256, 512],
        "PEPTIDE.NUM_FILTERS": [[64, 64, 64], [128, 128, 128], [256, 256, 256]],
        "PROTEIN.NUM_FILTERS": [[64, 64, 64], [128, 128, 128], [256, 256, 256]],
        "BCN.HEADS": (2, 8),
    }

    results_file = f'result/{args.data}_hy.csv'
    yaml_file = f'configs/{args.data}.yaml'

    all_results = []
    dataset_results = random_search(cfg, args.data, param_grid, results_file, num_samples=10, num_runs=3)
    all_results.append(dataset_results)

    # 分析结果
    analyze_results(results_file, cfg, yaml_file)