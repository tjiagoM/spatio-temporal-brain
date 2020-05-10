import argparse

import torch
import wandb
from torch_geometric.data import DataLoader

from datasets import HCPDataset
from main_loop import send_global_results, generate_st_model, evaluate_classifier
from model import SpatioTemporalModel
from utils import DatasetType, AnalysisType, ConnType, \
    create_name_for_model, Normalisation, create_name_for_brain_dataset, EncodingStrategy, PoolingStrategy, \
    ConvStrategy, SweepType

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

print('Args are', run_id, dropout, weight_d)

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

# Getting best model
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

# Getting HCP Data
name_dataset = create_name_for_brain_dataset(num_nodes=68,
                                             time_length=1200,
                                             target_var='gender',
                                             threshold=w_config['threshold'],
                                             normalisation=w_config['param_normalisation'],
                                             connectivity_type=w_config['param_conn_type'],
                                             analysis_type=w_config['analysis_type'],
                                             encoding_strategy=w_config['param_encoding_strategy'],
                                             dataset_type=DatasetType('hcp'))
print('Going with', name_dataset)
dataset = HCPDataset(root=name_dataset,
                     target_var='gender',
                     num_nodes=68,
                     threshold=w_config['threshold'],
                     connectivity_type=w_config['param_conn_type'],
                     normalisation=w_config['param_normalisation'],
                     analysis_type=w_config['analysis_type'],
                     encoding_strategy=w_config['param_encoding_strategy'],
                     time_length=1200)

# dataset.data is private, might change in future versions of pyg...
dataset.data.x = dataset.data.x[:, :490]

test_out_loader = DataLoader(dataset, batch_size=w_config['batch_size'], shuffle=False)
test_metrics = evaluate_classifier(model, test_out_loader, w_config['param_pooling'], w_config['device_run'])
print(test_metrics)

wandb.init(entity='st-team', name=f'ukb_eval_on_hcp_{run_id}')

send_global_results(test_metrics)
