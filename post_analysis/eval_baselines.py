import json
import os

import numpy as np

os.chdir('..')


def print_metrics(model_name, run_name):
    metrics = {'acc': [], 'auc': [], 'sensitivity': [], 'specificity': []}

    for fold_num in range(5):
        with open(f'results/xgb_{run_name}_test_{fold_num+1}.json', 'r') as read_file:
            data = json.load(read_file)

            for metric in metrics.keys():
                metrics[metric].append(data[metric])

    print(model_name, end=' & ')
    print(f'{np.mean(metrics["auc"]):.2f} ({np.std(metrics["auc"]):.3f}) & '
          f'{np.mean(metrics["acc"]):.2f} ({np.std(metrics["acc"]):.3f}) & '
          f'{np.mean(metrics["sensitivity"]):.2f} ({np.std(metrics["sensitivity"]):.3f}) & '
          f'{np.mean(metrics["specificity"]):.2f} ({np.std(metrics["specificity"]):.3f})')


if __name__ == '__main__':
    for run_name in ['hcp_st_unimodal', 'hcp_st_multimodal', 'ukb_st_unimodal']:
        print(run_name, ':')
        print_metrics('XGBoost', run_name)
