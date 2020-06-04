import numpy as np
import torch
import wandb
from torch_geometric.data import DataLoader

from datasets import HCPDataset
from main_loop import generate_st_model, evaluate_model
from model import SpatioTemporalModel
from utils import DatasetType, AnalysisType, ConnType, \
    create_name_for_model, Normalisation, create_name_for_brain_dataset, EncodingStrategy, PoolingStrategy, \
    ConvStrategy, SweepType

# Extra fields besides run_id are due to precision issues when saving.
best_runs = {
    'no_diffpool': {0: {'run_id': 'fdy5th0d', 'model_v': '64'},
                    1: {'run_id': 'w8ylfez9', 'dropout': 0.24218285663325959},
                    2: {'run_id': 'xl8woeqr'},
                    3: {'run_id': 'jmjga3w9', 'weight_d': 0.0072375354916992245},
                    4: {'run_id': 'aft5sncg'}},
    'no_mean': {0: {'run_id': 'ffy60yhy', 'model_v': '64'},
                1: {'run_id': '33tbqog2'},
                2: {'run_id': 'fepf04je', 'weight_d': 4.0287379093021184e-07},
                3: {'run_id': 'dhl9l0y4'},
                4: {'run_id': 'cijrrmgf'}},
    'n_e_mean': {0: {'run_id': '65slgxut', 'weight_d': 9.017758245804703e-06},
                 1: {'run_id': 'uqakiqlk'},
                 2: {'run_id': 'i9a83qtc'},
                 3: {'run_id': 'jp88x8mf', 'dropout': 0.025361867527413186},
                 4: {'run_id': 'q5te841d'}},
    'n_e_diffpool': {0: {'run_id': 'zuctoloq'},
                     1: {'run_id': '68fgmmdo'},
                     2: {'run_id': '7vb4ckzl', 'dropout': 0.20686909521891877},
                     3: {'run_id': 'mt29y65e'},
                     4: {'run_id': 'fjnpo77p'}},
    'node_mean': {0: {'run_id': '3yj09s2x'},
                  1: {'run_id': 'u6cfugyc'},
                  2: {'run_id': '7597akad'},
                  3: {'run_id': 'j63gkpoe'},
                  4: {'run_id': ''}},

    'node_diffpool': {0: {'run_id': '077bkvxp'},
                      1: {'run_id': '4tle1l3g', 'dropout': 0.22238630459171502},
                      2: {'run_id': '88eje3no'},
                      3: {'run_id': 'x94lygb9'},
                      4: {'run_id': 'yaogr549'}}
}

DEVICE_RUN = 'cuda:1'

for model_type, runs_all in best_runs.items():
    print('----', model_type)
    metrics_ukb = {'f1': [], 'acc': [], 'auc': [], 'sensitivity': [], 'specificity': []}
    metrics_hcp = {'f1': [], 'acc': [], 'auc': [], 'sensitivity': [], 'specificity': []}
    for fold_num, run_info in runs_all.items():
        run_id = run_info['run_id']
        # print('Args are', run_id, device_run, dropout, weight_d)

        api = wandb.Api()
        best_run = api.run(f'/st-team/spatio-temporal-brain/runs/{run_id}')
        # w_config = best_run.config
        for metric in metrics_ukb.keys():
            metrics_ukb[metric].append(best_run.summary[f'values_test_{metric}'])

        # Running for HCP
        w_config = best_run.config

        w_config['analysis_type'] = AnalysisType(w_config['analysis_type'])
        w_config['dataset_type'] = DatasetType(w_config['dataset_type'])
        w_config['device_run'] = DEVICE_RUN
        w_config['param_lr'] = w_config['lr']
        w_config['model_with_sigmoid'] = True
        w_config['param_activation'] = w_config['activation']
        w_config['param_channels_conv'] = w_config['channels_conv']
        w_config['param_conn_type'] = ConnType(w_config['conn_type'])
        w_config['param_conv_strategy'] = ConvStrategy(w_config['conv_strategy'])
        if 'dropout' not in run_info.keys():
            w_config['param_dropout'] = w_config['dropout']
        else:
            w_config['param_dropout'] = float(run_info['dropout'])
        w_config['param_encoding_strategy'] = EncodingStrategy(w_config['encoding_strategy'])
        w_config['param_normalisation'] = Normalisation(w_config['normalisation'])
        w_config['param_num_gnn_layers'] = w_config['num_gnn_layers']
        w_config['param_pooling'] = PoolingStrategy(w_config['pooling'])
        if 'weight_d' not in run_info.keys():
            w_config['param_weight_decay'] = w_config['weight_decay']
        else:
            w_config['param_weight_decay'] = float(run_info['weight_d'])

        w_config['sweep_type'] = SweepType(w_config['sweep_type'])
        w_config['param_gat_heads'] = 0
        if w_config['sweep_type'] == SweepType.GAT:
            w_config['param_gat_heads'] = w_config.gat_heads

        if w_config['analysis_type'] == AnalysisType.ST_MULTIMODAL:
            w_config['multimodal_size'] = 10
        elif w_config['analysis_type'] == AnalysisType.ST_UNIMODAL:
            w_config['multimodal_size'] = 0

        if w_config['target_var'] in ['age', 'bmi']:
            w_config['model_with_sigmoid'] = False

        # Getting best model
        inner_fold_for_val: int = 1
        model: SpatioTemporalModel = generate_st_model(w_config, for_test=True)
        if 'model_v' in run_info.keys():
            model.VERSION = run_info['model_v']
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
        if 'model_v' in run_info.keys():
            # We know the very specific "old" cases
            if w_config['param_pooling'] == PoolingStrategy.DIFFPOOL:
                model_saving_path = model_saving_path.replace('T_difW_F', 'GC_FGA_F')
            elif w_config['param_pooling'] == PoolingStrategy.MEAN:
                model_saving_path = model_saving_path.replace('T_no_W_F', 'GC_FGA_F')
        model.load_state_dict(torch.load(model_saving_path, map_location=w_config['device_run']))
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
                                                     dataset_type=DatasetType('hcp'),
                                                     edge_weights=w_config['edge_weights'])
        print('Going with', name_dataset)
        dataset = HCPDataset(root=name_dataset,
                             target_var='gender',
                             num_nodes=68,
                             threshold=w_config['threshold'],
                             connectivity_type=w_config['param_conn_type'],
                             normalisation=w_config['param_normalisation'],
                             analysis_type=w_config['analysis_type'],
                             encoding_strategy=w_config['param_encoding_strategy'],
                             time_length=1200,
                             edge_weights=w_config['edge_weights'])

        # dataset.data is private, might change in future versions of pyg...
        dataset.data.x = dataset.data.x[:, :490]

        test_out_loader = DataLoader(dataset, batch_size=w_config['batch_size'], shuffle=False)
        test_metrics = evaluate_model(model, test_out_loader, w_config['param_pooling'], w_config['device_run'])
        for metric in metrics_hcp.keys():
            metrics_hcp[metric].append(test_metrics[metric])

    print('UKB:')
    print(f'{round(np.mean(metrics_ukb["auc"]), 2)} ({round(np.std(metrics_ukb["auc"]), 3)}) & '
          f'{round(np.mean(metrics_ukb["acc"]), 2)} ({round(np.std(metrics_ukb["acc"]), 3)}) & '
          f'{round(np.mean(metrics_ukb["sensitivity"]), 2)} ({round(np.std(metrics_ukb["sensitivity"]), 3)}) & '
          f'{round(np.mean(metrics_ukb["specificity"]), 2)} ({round(np.std(metrics_ukb["specificity"]), 3)})')

    print('HCP:')
    print(f'{round(np.mean(metrics_hcp["auc"]), 2)} ({round(np.std(metrics_hcp["auc"]), 3)}) & '
          f'{round(np.mean(metrics_hcp["acc"]), 2)} ({round(np.std(metrics_hcp["acc"]), 3)}) & '
          f'{round(np.mean(metrics_hcp["sensitivity"]), 2)} ({round(np.std(metrics_hcp["sensitivity"]), 3)}) & '
          f'{round(np.mean(metrics_hcp["specificity"]), 2)} ({round(np.std(metrics_hcp["specificity"]), 3)})')
