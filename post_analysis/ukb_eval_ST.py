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
best_runs_ukb = {
    '100_n_diffpool': {0: {'run_id': 'khnljhrj'},
                       1: {'run_id': 'k9y54v5w', 'weight_d': 0.0012895162344404025},
                       2: {'run_id': '8ulilkox'},
                       3: {'run_id': 'm1lyyxez'},
                       4: {'run_id': 'lz7r38t4'}}
    # '100_n_mean': {0: {'run_id': 'zqfxsg2g'},
    #                    1: {'run_id': 'uiuoh583'},
    #                    2: {'run_id': 'lryrz1z8', 'weight_d': 2.5965696862532323e-07},
    #                    3: {'run_id': '937rms0w'},
    #                    4: {'run_id': 'sqn6ovck', 'dropout': 0.46961407058088156}},
    # '100_n_e_mean': {0: {'run_id': '94uhovir', 'weight_d': 1.5759293238676228e-07},
    #                    1: {'run_id': 'rag1ypk2', 'lr': 1.0061199415847143e-05, 'dropout': 0.09294885751683715},
    #                    2: {'run_id': '9l3dd2lh'},
    #                    3: {'run_id': 'iugiapic'},
    #                    4: {'run_id': 'ndp2mqm2', 'lr': 0.00011814265529647913}},
    # '100_n_e_diffpool': {0: {'run_id': '1oysy05q'},
    #                1: {'run_id': 'nxqb9kvj'},
    #                2: {'run_id': 'skripjyc', 'weight_d': 1.5483273684368499e-06},
    #                3: {'run_id': '6b3si6pc'},
    #                4: {'run_id': 's1nhqmnj'}},

    # 'no_diffpool': {0: {'run_id': 'fdy5th0d', 'model_v': '64'},
    #                1: {'run_id': 'w8ylfez9', 'dropout': 0.24218285663325959},
    #                2: {'run_id': 'xl8woeqr'},
    #                3: {'run_id': 'jmjga3w9', 'weight_d': 0.0072375354916992245},
    #                4: {'run_id': 'aft5sncg'}},
    # 'no_mean': {0: {'run_id': 'ffy60yhy', 'model_v': '64'},
    #            1: {'run_id': '33tbqog2'},
    #            2: {'run_id': 'fepf04je', 'weight_d': 4.0287379093021184e-07},
    #            3: {'run_id': 'dhl9l0y4'},
    #            4: {'run_id': 'cijrrmgf'}},
    # 'n_e_mean': {0: {'run_id': '65slgxut', 'weight_d': 9.017758245804703e-06},
    #             1: {'run_id': 'uqakiqlk'},
    #             2: {'run_id': 'i9a83qtc'},
    #             3: {'run_id': 'jp88x8mf', 'dropout': 0.025361867527413186},
    #             4: {'run_id': 'q5te841d'}},
    # 'n_e_diffpool': {0: {'run_id': 'zuctoloq'},
    #                 1: {'run_id': '68fgmmdo'},
    #                 2: {'run_id': '7vb4ckzl', 'dropout': 0.20686909521891877},
    #                 3: {'run_id': 'mt29y65e'},
    #                 4: {'run_id': 'fjnpo77p'}},
    # 'node_mean': {0: {'run_id': '3yj09s2x'},
    #              1: {'run_id': 'u6cfugyc'},
    #              2: {'run_id': '7597akad'},
    #              3: {'run_id': 'j63gkpoe'},
    #              4: {'run_id': 'l1w81fh2', 'dropout': 0.37245464912261556, 'lr': 0.00048327684496385213}},
    # 'node_diffpool': {0: {'run_id': '077bkvxp'},
    #                  1: {'run_id': '4tle1l3g', 'dropout': 0.22238630459171502},
    #                  2: {'run_id': '88eje3no'},
    #                  3: {'run_id': 'x94lygb9'},
    #                  4: {'run_id': 'yaogr549'}}
}

best_runs_hcp = {
    'N + E $\\rightarrow$ Average': {0: {'run_id': 'sa4vz637'},
                                     1: {'run_id': 'qvc3red3'},
                                     2: {'run_id': 'ho5ei0bu', 'weight_d': 0.00045494145445063716},
                                     3: {'run_id': '4gh6r613', 'dropout': 0.18552011553500802},
                                     4: {'run_id': 'sa4vz637'}},

    'N + E $\\rightarrow$ DiffPool': {0: {'run_id': 'rjqgjo6t', 'lr': 0.0073454738206698645},
                                      1: {'run_id': 'qhe3tmpg', 'weight_d': 0.0009706058274764595},
                                      2: {'run_id': 'w8zrimq3', 'weight_d': 1.6485231846614363e-07},
                                      3: {'run_id': 'e3hwsnkt', 'weight_d': 0.001836297034235979, 'dropout': 0.45716370274856044},
                                      4: {'run_id': 'fsn6xi4c'}},

    'N $\\rightarrow$ Average': {0: {'run_id': 'kb0h0cqg', 'weight_d': 2.3628232603528274e-05},
                                 1: {'run_id': 'jq3p5ywv'},
                                 2: {'run_id': 'ru0v7dw2', 'weight_d': 1.8440482416802742e-06},
                                 3: {'run_id': 'wkuh266s'},
                                 4: {'run_id': 'ncospvb1', 'lr': 0.0009052654172784859}},

    'N $\\rightarrow$ DiffPool': {0: {'run_id': '0rmhut2t'},
                                  1: {'run_id': '2qx0o67s'},
                                  2: {'run_id': 'ldmx3h4m', 'dropout': 0.43977735162063747, 'weight_d': 0.00010610234153585897},
                                  3: {'run_id': 'lhw0byv5'},
                                  4: {'run_id': 'r2r6wedp'}},

    '$\\rightarrow$ DiffPool': {0: {'run_id': 'vzz9dp8c'},
                                1: {'run_id': 'u92k3gck', 'dropout': 0.44280658914412885},
                                2: {'run_id': 'mj3u34ae'},
                                3: {'run_id': 'h9ltbjhh'},
                                4: {'run_id': 'xu3w7v5o'}},

    '$\\rightarrow$ Average': {0: {'run_id': 'u3g615fx', 'dropout': 0.09824567099751609},
                               1: {'run_id': 'i2t60lbt'},
                               2: {'run_id': '0reqccd2'},
                               3: {'run_id': 'fozuxf7z', 'dropout': 0.10414535975968037, 'weight_d': 0.0010280190032795579},
                               4: {'run_id': 'sqz196tb', 'dropout': 0.23326601015409346}}
}

best_runs_hcp100_THRES = {
    'N + E $\\rightarrow$ Average': {0: {'run_id': 'n01iu8xy'},
                                     1: {'run_id': 'fanpxslb'},
                                     2: {'run_id': '03tubetj'},
                                     3: {'run_id': 'm2uvk64e'},
                                     4: {'run_id': 'z3y05mah'}},

    'N + E $\\rightarrow$ DiffPool': {0: {'run_id': 'a9rmzt3g', 'weight_d': 9.811306344339555e-07},
                                      1: {'run_id': 'r493ayys'},
                                      2: {'run_id': 'wziewuzd'},
                                      3: {'run_id': 'd2w9v9gs'},
                                      4: {'run_id': 't2r06tnx', 'dropout': 0.37409969176335156}},

    'N $\\rightarrow$ Average': {0: {'run_id': '1es5sh3n', 'weight_d': 1.8837375147749088e-07},
                                 1: {'run_id': 'ltbs42t0'},
                                 2: {'run_id': '176rklv8'},
                                 3: {'run_id': '5zllqu19'},
                                 4: {'run_id': '7i5l5401', 'lr': 0.00019906466949509087}},

    'N $\\rightarrow$ DiffPool': {0: {'run_id': 'okwt3li5', 'weight_d': 1.2414858556355677e-05},
                                  1: {'run_id': 'iitaqcnl', 'lr': 0.00036632425868720524},
                                  2: {'run_id': '0yt9maap'},
                                  3: {'run_id': '18uo4s8c'},
                                  4: {'run_id': 'w0h1kykp', 'lr': 0.0012616677803190923}}
}

DEVICE_RUN = 'cpu'


def print_metrics(model_name, runs_all, validate_hcp=False):
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
        if 'lr' not in run_info.keys():
            w_config['param_lr'] = w_config['lr']
        else:
            w_config['param_lr'] = float(run_info['lr'])
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
        if not validate_hcp:
            continue
        else:
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

    # print('UKB:')
    print(model_name, end=' & ')
    print(f'{round(np.mean(metrics_ukb["auc"]), 2)} ({round(np.std(metrics_ukb["auc"]), 3)}) & '
          f'{round(np.mean(metrics_ukb["acc"]), 2)} ({round(np.std(metrics_ukb["acc"]), 3)}) & '
          f'{round(np.mean(metrics_ukb["sensitivity"]), 2)} ({round(np.std(metrics_ukb["sensitivity"]), 3)}) & '
          f'{round(np.mean(metrics_ukb["specificity"]), 2)} ({round(np.std(metrics_ukb["specificity"]), 3)})')

    if validate_hcp:
        print('HCP:')
        print(f'{round(np.mean(metrics_hcp["auc"]), 2)} ({round(np.std(metrics_hcp["auc"]), 3)}) & '
              f'{round(np.mean(metrics_hcp["acc"]), 2)} ({round(np.std(metrics_hcp["acc"]), 3)}) & '
              f'{round(np.mean(metrics_hcp["sensitivity"]), 2)} ({round(np.std(metrics_hcp["sensitivity"]), 3)}) & '
              f'{round(np.mean(metrics_hcp["specificity"]), 2)} ({round(np.std(metrics_hcp["specificity"]), 3)})')


if __name__ == '__main__':

    #for model_type, runs_all in best_runs_hcp.items():
    for model_type, runs_all in best_runs_hcp100_THRES.items():
        # print('----', model_type)
        print_metrics(model_type, runs_all)
