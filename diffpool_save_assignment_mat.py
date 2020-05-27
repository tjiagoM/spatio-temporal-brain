import argparse

import numpy as np
import torch
import torch_geometric.utils as pyg_utils
import wandb
from torch_geometric.data import DataLoader

from datasets import BrainDataset
from main_loop import generate_dataset, create_fold_generator, generate_st_model
from model import SpatioTemporalModel
from utils import Normalisation, ConnType, DatasetType, \
    AnalysisType, EncodingStrategy, ConvStrategy, PoolingStrategy, SweepType, create_name_for_model

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', help='run id')
parser.add_argument('--device', help='device to put the model', default='cuda:1')
parser.add_argument('--dropout', help='in case there was a precision problem')
parser.add_argument('--weight_d', help='in case of precision problems')
parser.add_argument('--model_v', help='in case of older models')
args = parser.parse_args()

run_id = args.run_id
device_run = args.device
dropout = args.dropout
weight_d = args.weight_d
model_v = args.model_v

print('Args are', run_id, device_run, dropout, weight_d, model_v)

# To save info about some runs
# no-> diffpool:
# run_id = 'fdy5th0d'
# model_v = '64'
# node -> diffpool:
# --run_id 077bkvxp
# nodeedge -> diffpool:
# --run_id zuctoloq

# For testing
# device_run = 'cuda:1'
# run_id = 'fdy5th0d'
# model_v = '64'
# dropout = None
# weight_d = None

api = wandb.Api()
best_run = api.run(f'/st-team/spatio-temporal-brain/runs/{run_id}')
w_config = best_run.config

w_config['analysis_type'] = AnalysisType(w_config['analysis_type'])
w_config['dataset_type'] = DatasetType(w_config['dataset_type'])
w_config['device_run'] = device_run
w_config['param_lr'] = w_config['lr']
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
if weight_d is None:
    w_config['param_weight_decay'] = w_config['weight_decay']
else:
    w_config['param_weight_decay'] = float(weight_d)

w_config['sweep_type'] = SweepType(w_config['sweep_type'])
w_config['param_gat_heads'] = 0
if w_config['sweep_type'] == SweepType.GAT:
    w_config['param_gat_heads'] = w_config.gat_heads

w_config['param_threshold'] = w_config['threshold']

if w_config['analysis_type'] == AnalysisType.ST_MULTIMODAL:
    w_config['multimodal_size'] = 10
elif w_config['analysis_type'] == AnalysisType.ST_UNIMODAL:
    w_config['multimodal_size'] = 0

if w_config['target_var'] in ['age', 'bmi']:
    w_config['model_with_sigmoid'] = False

dataset: BrainDataset = generate_dataset(w_config)

N_OUT_SPLITS = 5
N_INNER_SPLITS = 5

skf_outer_generator = create_fold_generator(dataset, w_config, N_OUT_SPLITS)

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
if model_v is not None:
    model.VERSION = model_v
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
                                               lr=w_config['param_lr'],
                                               weight_decay=w_config['param_weight_decay'],
                                               edge_weights=w_config['edge_weights'])
if model_v is not None:
    # We know the very specific "old" cases
    if w_config['param_pooling'] == PoolingStrategy.DIFFPOOL:
        model_saving_path = model_saving_path.replace('T_difW_F', 'GC_FGA_F')
model.load_state_dict(torch.load(model_saving_path, map_location=w_config['device_run']))
model.eval()

# Calculating on test set
# needs cast to int() because of higher precision when reading the csv
test_out_loader = DataLoader(X_test_out, batch_size=int(w_config['batch_size']), shuffle=False)

# TODO: change forward() function in model for this to be more straightforward
in_index = 0
NEW_MAX_NODES = 17
clusters = {
    0: torch.zeros((68, 68)).to(device_run),
    1: torch.zeros((68, 68)).to(device_run)
}
total_amount = {
    0: 0,
    1: 0
}

for data in test_out_loader:
    with torch.no_grad():
        data = data.to(device_run)
        print("Try 1")
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if model.multimodal_size > 0:
            xn, x = x[:, :model.multimodal_size], x[:, model.multimodal_size:]
            xn = model.multimodal_lin(xn)
            xn = model.activation(xn)
            xn = model.multimodal_batch(xn)
            # xn = F.dropout(xn, p=self.dropout, training=self.training)

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

        if model.sweep_type in [SweepType.META_NODE, SweepType.META_EDGE_NODE]:
            x, edge_attr, _ = model.meta_layer(x, edge_index, edge_attr)
        print("Try 2")
        adj_tmp = pyg_utils.to_dense_adj(edge_index, data.batch, edge_attr=edge_attr)
        if edge_attr is not None:  # Because edge_attr only has 1 feature per edge
            adj_tmp = adj_tmp[:, :, :, 0]
        x_tmp, batch_mask = pyg_utils.to_dense_batch(x, data.batch)
        # x, link_loss, ent_loss = model.diff_pool(x_tmp, adj_tmp, batch_mask)
        print("Try 2")
        s = model.diff_pool.gnn1_pool(x_tmp, adj_tmp, batch_mask)
        s = s.unsqueeze(0) if s.dim() == 2 else s
        s = torch.softmax(s, dim=-1)

        # Going over each assigment matrix of the batch
        for batch_i in range(s.shape[0]):
            sex_info = data.y[batch_i].item()
            total_amount[sex_info] += 1
            for col in range(NEW_MAX_NODES):
                matches = (s[batch_i][:, col] > 0.5).nonzero()
                # Going over each match
                for elem_i in range(matches.shape[0]):
                    for elem_j in range(elem_i + 1, matches.shape[0]):
                        clusters[sex_info][matches[elem_i].item(), matches[elem_j].item()] += 1
                        # print(matches[elem_i].item(), matches[elem_j].item())
        # break
        # np.save(f'diffpool_interp/s2_tmp{in_index}.npy', s.detach().cpu().numpy())
        # in_index += 1

np.save(f'results/dp_interp_{run_id}_male.npy', clusters[1].numpy())
np.save(f'results/dp_interp_{run_id}_female.npy', clusters[0].numpy())
a_female = clusters[0].numpy()
a_male = clusters[1].numpy()
