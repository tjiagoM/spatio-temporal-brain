from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
from torch_geometric.data import DataLoader

from datasets import UKBDataset, HCPDataset, BrainDataset
from main_loop import generate_dataset, create_fold_generator, generate_st_model
from model import SpatioTemporalModel
from utils import create_name_for_brain_dataset, Normalisation, ConnType, StratifiedGroupKFold, DatasetType, \
    AnalysisType, EncodingStrategy, ConvStrategy, PoolingStrategy, SweepType, create_name_for_model

best_model_name = 'best_models_hyperparams/best_st_hcp_multi_gender_1_struct_none_diff_pool.csv'

info_best = pd.read_csv(best_model_name, float_precision='round_trip')

device = 'cpu'

# Getting all configurations
run_cfg: Dict[str, Any] = {
        'analysis_type': AnalysisType(info_best.loc[0, 'analysis_type']),
        'batch_size': info_best.loc[0, 'batch_size'],
        'dataset_type': DatasetType(info_best.loc[0, 'dataset_type']),
        'device_run': device,
        'early_stop_steps': info_best.loc[0, 'early_stop_steps'],
        'model_with_sigmoid': True,
        'num_epochs': info_best.loc[0, 'num_epochs'],
        'num_nodes': info_best.loc[0, 'num_nodes'],
        'param_activation': info_best.loc[0, 'activation'],
        'param_channels_conv': info_best.loc[0, 'channels_conv'],
        'param_conn_type': ConnType(info_best.loc[0, 'conn_type']),
        'param_conv_strategy': ConvStrategy(info_best.loc[0, 'conv_strategy']),
        'param_dropout': info_best.loc[0, 'dropout'],
        'param_encoding_strategy': EncodingStrategy(info_best.loc[0, 'encoding_strategy']),
        'param_lr': info_best.loc[0, 'lr'],
        'param_normalisation': Normalisation(info_best.loc[0, 'normalisation']),
        'param_num_gnn_layers': info_best.loc[0, 'num_gnn_layers'],
        'param_threshold': info_best.loc[0, 'threshold'],
        'param_weight_decay': info_best.loc[0, 'weight_decay'],
        'split_to_test': info_best.loc[0, 'fold_num'],
        'target_var': info_best.loc[0, 'target_var'],
        'time_length': info_best.loc[0, 'time_length'],
        'param_pooling': PoolingStrategy(info_best.loc[0, 'pooling'])
    }

sweep_type = SweepType(info_best.loc[0, 'sweep_type'])
run_cfg['param_gat_heads'] = 0
run_cfg['param_add_gcn'] = False
run_cfg['param_add_gat'] = False
if sweep_type == SweepType.GCN:
    run_cfg['param_add_gcn'] = True
elif sweep_type == SweepType.GAT:
    run_cfg['param_add_gat'] = True
    run_cfg['param_gat_heads'] = info_best.loc[0, 'gat_heads']

if run_cfg['analysis_type'] == AnalysisType.ST_MULTIMODAL:
    run_cfg['multimodal_size'] = 10
elif run_cfg['analysis_type'] == AnalysisType.ST_UNIMODAL:
    run_cfg['multimodal_size'] = 0

dataset: BrainDataset = generate_dataset(run_cfg)

N_OUT_SPLITS = 5
N_INNER_SPLITS = 5

skf_outer_generator = create_fold_generator(dataset, run_cfg['dataset_type'], N_OUT_SPLITS)


# Getting train / test folds
outer_split_num: int = 0
for train_index, test_index in skf_outer_generator:
    outer_split_num += 1
    # Only run for the specific fold defined in the script arguments.
    if outer_split_num != run_cfg['split_to_test']:
        continue

    X_test_out = dataset[torch.tensor(test_index)]

    break


inner_fold_for_val: int = 1
model: SpatioTemporalModel = generate_st_model(run_cfg, for_test=True)
model_saving_path: str = create_name_for_model(target_var=run_cfg['target_var'],
                                               model=model,
                                               outer_split_num=run_cfg['split_to_test'],
                                               inner_split_num=inner_fold_for_val,
                                               n_epochs=run_cfg['num_epochs'],
                                               threshold=run_cfg['param_threshold'],
                                               batch_size=run_cfg['batch_size'],
                                               num_nodes=run_cfg['num_nodes'],
                                               conn_type=run_cfg['param_conn_type'],
                                               normalisation=run_cfg['param_normalisation'],
                                               analysis_type=run_cfg['analysis_type'],
                                               metric_evaluated='loss',
                                               dataset_type=run_cfg['dataset_type'],
                                               lr=run_cfg['param_lr'],
                                               weight_decay=run_cfg['param_weight_decay'])
model.load_state_dict(torch.load(model_saving_path))
model.eval()

# Calculating on test set
# needs cast to int() because of higher precision when reading the csv
test_out_loader = DataLoader(X_test_out, batch_size=int(run_cfg['batch_size']), shuffle=False)

# TODO: change forward function for this to be more straightforward
in_index = 0
for data in test_out_loader:
    with torch.no_grad():
        data = data.to(device)
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
        np.save(f'diffpool_interp/s2_tmp{in_index}.npy', s.detach().cpu().numpy())
        in_index += 1
