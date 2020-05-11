from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
import wandb
from torch_geometric.data import DataLoader
import argparse

from datasets import UKBDataset, HCPDataset, BrainDataset
from main_loop import generate_dataset, create_fold_generator, generate_st_model
from model import SpatioTemporalModel
from utils import create_name_for_brain_dataset, Normalisation, ConnType, StratifiedGroupKFold, DatasetType, \
    AnalysisType, EncodingStrategy, ConvStrategy, PoolingStrategy, SweepType, create_name_for_model

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', help='run id')
parser.add_argument('--device', help='device to put the model', default='cuda:1')
parser.add_argument('--dropout', help='in case there was a precision problem')
parser.add_argument('--weight_d', help='in case of precision problems')
args = parser.parse_args()

run_id = args.run_id
device_run = args.device
dropout = args.dropout
weight_d = args.weight_d

device_run = 'cuda:1'
run_id = 'zj20jsla'

api = wandb.Api()
best_run = api.run(f'/st-team/spatio-temporal-brain/runs/{run_id}')
w_config = best_run.config

w_config['analysis_type'] = AnalysisType(w_config['analysis_type'])
w_config['dataset_type'] = DatasetType(w_config['dataset_type'])
w_config['device_run'] = device_run
w_config['model_with_sigmoid'] = True
w_config['param_activation'] = w_config['activation']
w_config['param_channels_conv'] = w_config['channels_conv']
w_config['param_conn_type'] = ConnType(w_config['conn_type'])
w_config['param_conv_strategy'] = ConvStrategy(w_config['conv_strategy'])
if dropout is None:
    w_config['param_dropout'] = w_config['dropout']
else:
    w_config['param_dropout'] = float(dropout)
w_config['param_encoding_strategy'] = EncodingStrategy(w_config['encoding_strategy'])
w_config['param_normalisation'] = Normalisation(w_config['normalisation'])
w_config['param_num_gnn_layers'] = w_config['num_gnn_layers']
w_config['param_pooling'] = PoolingStrategy(w_config['pooling'])
w_config['param_threshold'] = w_config['threshold']
if weight_d is not None:
    w_config['weight_decay'] = float(weight_d)

sweep_type = SweepType(w_config['sweep_type'])
w_config['param_gat_heads'] = 0
w_config['param_add_gcn'] = False
w_config['param_add_gat'] = False
if sweep_type == SweepType.GCN:
    w_config['param_add_gcn'] = True
elif sweep_type == SweepType.GAT:
    w_config['param_add_gat'] = True
    w_config['param_gat_heads'] = w_config['gat_heads']
if w_config['param_pooling'] == PoolingStrategy.CONCAT:
    w_config['batch_size'] -= 50
if w_config['analysis_type'] == AnalysisType.ST_MULTIMODAL:
    w_config['multimodal_size'] = 10
elif w_config['analysis_type'] == AnalysisType.ST_UNIMODAL:
    w_config['multimodal_size'] = 0


dataset: BrainDataset = generate_dataset(w_config)

N_OUT_SPLITS = 5
N_INNER_SPLITS = 5

skf_outer_generator = create_fold_generator(dataset, w_config['dataset_type'], w_config['analysis_type'], N_OUT_SPLITS)


# Getting train / test folds
outer_split_num: int = 0
for train_index, test_index in skf_outer_generator:
    outer_split_num += 1
    # Only run for the specific fold defined in the script arguments.
    if outer_split_num != w_config['fold_num']:
        continue

    X_test_out = dataset[torch.tensor(test_index)]

    break


inner_fold_for_val: int = 1
model: SpatioTemporalModel = generate_st_model(w_config, for_test=True)
model_saving_path: str = create_name_for_model(target_var=w_config['target_var'],
                                               model=model,
                                               outer_split_num=w_config['fold_num'],
                                               inner_split_num=inner_fold_for_val,
                                               n_epochs=w_config['num_epochs'],
                                               threshold=w_config['threshold'],
                                               batch_size=w_config['batch_size'],
                                               num_nodes=w_config['num_nodes'],
                                               conn_type=w_config['param_conn_type'],
                                               normalisation=w_config['param_normalisation'],
                                               analysis_type=w_config['analysis_type'],
                                               metric_evaluated='loss',
                                               dataset_type=w_config['dataset_type'],
                                               lr=w_config['lr'],
                                               weight_decay=w_config['weight_decay'])
model.load_state_dict(torch.load(model_saving_path))
model.eval()

# Calculating on test set
# needs cast to int() because of higher precision when reading the csv
test_out_loader = DataLoader(X_test_out, batch_size=int(w_config['batch_size']), shuffle=False)

# TODO: change forward function for this to be more straightforward
in_index = 0
for data in test_out_loader:
    with torch.no_grad():
        data = data.to(device_run)
        print("Try 1")
        x, edge_index = data.x, data.edge_index
        if model.multimodal_size > 0:
            xn, x = x[:, :model.multimodal_size], x[:, model.multimodal_size:]
            xn = model.multimodal_lin(xn)
            xn = model.activation(xn)
            xn = model.multimodal_batch(xn)
            #xn = F.dropout(xn, p=model.dropout, training=model.training)

        # Processing temporal part
        if model.conv_strategy != ConvStrategy.NONE:
            x = x.view(-1, 1, model.num_time_length)
            x = model.temporal_conv(x)

            # Concatenating for the final embedding per node
            x = x.view(-1, model.size_before_lin_temporal)
            x = model.lin_temporal(x)
            x = model.activation(x)
        elif model.encoding_strategy == EncodingStrategy.STATS:
            x = model.stats_lin(x)
            x = model.activation(x)
            x = model.stats_batch(x)

        if model.multimodal_size > 0:
            x = torch.cat((xn, x), dim=1)
        print("Try 2")
        adj_tmp = pyg_utils.to_dense_adj(edge_index, data.batch)
        x_tmp, batch_mask = pyg_utils.to_dense_batch(x, data.batch)
        # x, link_loss, ent_loss = model.diff_pool(x_tmp, adj_tmp, batch_mask)
        print("Try 2")
        s = model.diff_pool.gnn1_pool(x_tmp, adj_tmp, batch_mask)
        s = s.unsqueeze(0) if s.dim() == 2 else s
        s = torch.softmax(s, dim=-1)
        break
        np.save(f'diffpool_interp/s2_tmp{in_index}.npy', s.detach().cpu().numpy())
        in_index += 1
