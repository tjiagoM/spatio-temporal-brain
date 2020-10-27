import pickle

import numpy as np
import wandb
from xgboost import XGBModel

from datasets import FlattenCorrsDataset
from main_loop import generate_xgb_model, return_classifier_metrics
from utils import DatasetType, AnalysisType, ConnType, create_name_for_flattencorrs_dataset, create_name_for_xgbmodel

best_runs = {
    'sex_flatten': {0: {'run_id': 'zqahc9zb'},
                    1: {'run_id': 'vvhqjyhm'},
                    2: {'run_id': 'areerly7', 'ode': 0.9801913876776511},
                    3: {'run_id': '1o9rbbgd'},
                    4: {'run_id': 'igl3va7i'}}
}

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
        w_config['param_conn_type'] = ConnType(w_config['conn_type'])
        if 'ode' in run_info.keys():
            w_config['colsample_bynode'] = float(run_info['ode'])

        # Getting best model
        inner_fold_for_val: int = 1
        model: XGBModel = generate_xgb_model(w_config)
        model_saving_path = create_name_for_xgbmodel(model=model,
                                                     outer_split_num=w_config['fold_num'],
                                                     inner_split_num=inner_fold_for_val,
                                                     run_cfg=w_config
                                                     )
        model = pickle.load(open(model_saving_path, "rb"))

        # Getting HCP Data
        hcp_dict = {
            'dataset_type': DatasetType('hcp'),
            'analysis_type': AnalysisType('flatten_corrs'),
            'param_conn_type': ConnType('fmri'),
            'num_nodes': 68,
            'time_length': 1200
        }
        name_dataset = create_name_for_flattencorrs_dataset(hcp_dict)
        dataset = FlattenCorrsDataset(root=name_dataset,
                                      num_nodes=68,
                                      connectivity_type=ConnType('fmri'),
                                      analysis_type=AnalysisType('flatten_corrs'),
                                      dataset_type=DatasetType('hcp'),
                                      time_length=1200)
        hcp_arr = np.array([data.x.numpy() for data in dataset])
        hcp_y_test = [int(data.sex.item()) for data in dataset]

        test_metrics = return_classifier_metrics(hcp_y_test,
                                                 pred_prob=model.predict_proba(hcp_arr)[:, 1],
                                                 pred_binary=model.predict(hcp_arr),
                                                 flatten_approach=True)
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
