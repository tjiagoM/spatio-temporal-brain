import os
import pickle
import random
from collections import deque
from sys import exit
from typing import Dict, Any, Union
from torch_geometric.utils import degree
import numpy as np
import pandas as pd
import torch
import wandb
from scipy.stats import stats
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch_geometric.data import DataLoader
from xgboost import XGBClassifier, XGBRegressor, XGBModel

from datasets import BrainDataset, HCPDataset, UKBDataset, FlattenCorrsDataset
from main_loop import generate_dataset, create_fold_generator, get_empty_metrics_dict, generate_st_model, training_step
from model import SpatioTemporalModel
from utils import create_name_for_brain_dataset, create_name_for_model, Normalisation, ConnType, ConvStrategy, \
    StratifiedGroupKFold, PoolingStrategy, AnalysisType, merge_y_and_others, EncodingStrategy, create_best_encoder_name, \
    SweepType, DatasetType, get_freer_gpu, free_gpu_info, create_name_for_flattencorrs_dataset, \
    create_name_for_xgbmodel, calculate_indegree_histogram

wandb.init(project='st_extra')

run_cfg: Dict[str, Any] = {
    'analysis_type': AnalysisType('st_unimodal'),
    'dataset_type': DatasetType('ukb'),
    'num_nodes': 68,
    'param_conn_type': ConnType('fmri'),
    'split_to_test': 2,
    'target_var': 'gender',
    'time_length': 490,
}
if run_cfg['analysis_type'] in [AnalysisType.ST_UNIMODAL, AnalysisType.ST_MULTIMODAL, AnalysisType.ST_UNIMODAL_AVG, AnalysisType.ST_MULTIMODAL_AVG]:
    run_cfg['batch_size'] = 150
    run_cfg['device_run'] = f'cuda:{get_freer_gpu()}'
    run_cfg['early_stop_steps'] = 33
    run_cfg['edge_weights'] = True
    run_cfg['model_with_sigmoid'] = True
    run_cfg['num_epochs'] = 150
    run_cfg['param_activation'] = 'relu'
    run_cfg['param_channels_conv'] = 8
    run_cfg['param_conv_strategy'] = ConvStrategy('tcn_entire')
    run_cfg['param_dropout'] = 0.1
    run_cfg['param_encoding_strategy'] = EncodingStrategy('none')
    run_cfg['param_lr'] = 4.2791529866e-06
    run_cfg['param_normalisation'] = Normalisation('subject_norm')
    run_cfg['param_num_gnn_layers'] = 1
    run_cfg['param_pooling'] = PoolingStrategy('concat')
    run_cfg['param_threshold'] = 10
    run_cfg['param_weight_decay'] = 0.046926
    run_cfg['sweep_type'] = SweepType('node_meta')
    run_cfg['temporal_embed_size'] = 16

    run_cfg['ts_spit_num'] = int(4800 / run_cfg['time_length'])

    # Not sure whether this makes a difference with the cuda random issues, but it was in the examples :(
    #kwargs_dataloader = {'num_workers': 1, 'pin_memory': True} if run_cfg['device_run'].startswith('cuda') else {}

    # Definitions depending on sweep_type
    run_cfg['param_gat_heads'] = 0
    #if run_cfg['sweep_type'] == SweepType.GAT:
    #    run_cfg['param_gat_heads'] = config.gat_heads

    run_cfg['tcn_depth'] = 3
    run_cfg['tcn_kernel'] = 7
    run_cfg['tcn_hidden_units'] = 8
    run_cfg['tcn_final_transform_layers'] = 1
    run_cfg['tcn_norm_strategy'] = 'batchnorm'

    run_cfg['nodemodel_aggr'] = 'all'
    run_cfg['nodemodel_scalers'] = 'none'
    run_cfg['nodemodel_layers'] = 3
    run_cfg['final_mlp_layers'] = 1

N_OUT_SPLITS: int = 5
N_INNER_SPLITS: int = 5
run_cfg['multimodal_size'] = 0
print('Resulting run_cfg:', run_cfg)

dataset = generate_dataset(run_cfg)
skf_outer_generator = create_fold_generator(dataset, run_cfg, N_OUT_SPLITS)
outer_split_num: int = 0
for train_index, test_index in skf_outer_generator:
    outer_split_num += 1
    # Only run for the specific fold defined in the script arguments.
    if outer_split_num != run_cfg['split_to_test']:
        continue

    X_train_out = dataset[torch.tensor(train_index)]
    X_test_out = dataset[torch.tensor(test_index)]

    break

skf_inner_generator = create_fold_generator(X_train_out, run_cfg, N_INNER_SPLITS)
overall_metrics: Dict[str, list] = get_empty_metrics_dict(run_cfg)
inner_loop_run: int = 0
for inner_train_index, inner_val_index in skf_inner_generator:
    inner_loop_run += 1

    X_train_in = X_train_out[torch.tensor(inner_train_index)]
    X_val_in = X_train_out[torch.tensor(inner_val_index)]

    run_cfg['dataset_indegree'] = calculate_indegree_histogram(X_train_in)

    model: SpatioTemporalModel = generate_st_model(run_cfg)

    break

# Number of trainable params: 164597203 (more complex mean, 8 hidden units)
# Number of trainable params: 164618923 (more complex mean, 32 hidden units)
# Number of trainable params: 164795453 (more complex DP, 32 hidden)
#############################################################
############################################################
out_fold_num=run_cfg['split_to_test']
in_fold_num=inner_loop_run
run_cfg=run_cfg
model=model
X_train_in=X_train_in
X_val_in=X_val_in
label_scaler=None

train_in_loader = DataLoader(X_train_in, batch_size=run_cfg['batch_size'], shuffle=True)
val_loader = DataLoader(X_val_in, batch_size=run_cfg['batch_size'], shuffle=False)

for data in train_in_loader:
    data = data.to(run_cfg['device_run'])
    print(model(data).shape)
    break
