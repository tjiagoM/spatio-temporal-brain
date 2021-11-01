import numpy as np
import wandb

best_runs_tmp = {
    'tmp': {0: {'run_id': ''},
            1: {'run_id': ''},
            2: {'run_id': ''},
            3: {'run_id': ''},
            4: {'run_id': ''}},
}

best_runs_ukb = {
    #'N + E $\\rightarrow$ Concat': {0: {'run_id': ''},
    #                                1: {'run_id': ''},
    #                                2: {'run_id': ''},
    #                                3: {'run_id': ''},
    #                                4: {'run_id': ''}},
    # 'N + E $\\rightarrow$ DiffPool': {0: {'run_id': ''},
    #                    1: {'run_id': ''},
    #                    2: {'run_id': ''},
    #                    3: {'run_id': ''},
    #                    4: {'run_id': ''}},
    'N $\\rightarrow$ Concat': {0: {'run_id': '5ra9icg7'},
                                1: {'run_id': 'f2n8s6gw'},
                                2: {'run_id': '3vunou40'},
                                3: {'run_id': 'q6dl8u6b'},
                                4: {'run_id': '90e7orj6'}},
    # 'N $\\rightarrow$ DiffPool': {0: {'run_id': ''},
    #                    1: {'run_id': ''},
    #                    2: {'run_id': ''},
    #                    3: {'run_id': ''},
    #                    4: {'run_id': ''}},
}

best_runs_hcp_fmri = {
     'N + E $\\rightarrow$ Concat': {0: {'run_id': 'tdztbchl'},
                       1: {'run_id': '5t0e0oel'},
                       2: {'run_id': '6d5qb8c7'},
                       3: {'run_id': 'pw7h3kub'},
                       4: {'run_id': '4txocl2p'}},
    # 'N + E $\\rightarrow$ DiffPool': {0: {'run_id': ''},
    #                    1: {'run_id': ''},
    #                    2: {'run_id': ''},
    #                    3: {'run_id': ''},
    #                    4: {'run_id': ''}},
    'N $\\rightarrow$ Concat': {0: {'run_id': 'hovygim0'},
                                1: {'run_id': '3cej2yzp'},
                                2: {'run_id': 'gmfaxo2e'},
                                3: {'run_id': 'fv4xi5ve'},
                                4: {'run_id': 'c5hvjkq8'}},
    # 'N $\\rightarrow$ DiffPool': {0: {'run_id': ''},
    #                    1: {'run_id': ''},
    #                    2: {'run_id': ''},
    #                    3: {'run_id': ''},
    #                    4: {'run_id': ''}},
}

best_runs_hcp_struct = {
    #'N + E $\\rightarrow$ Concat': {0: {'run_id': ''},
    #                                1: {'run_id': ''},
    #                                2: {'run_id': ''},
    #                                3: {'run_id': ''},
    #                                4: {'run_id': ''}},
    # 'N + E $\\rightarrow$ DiffPool': {0: {'run_id': ''},
    #                    1: {'run_id': ''},
    #                    2: {'run_id': ''},
    #                    3: {'run_id': ''},
    #                    4: {'run_id': ''}},
    'N $\\rightarrow$ Concat': {0: {'run_id': 'xfjrfwmv'},
                                1: {'run_id': 'luzbl8k9'},
                                2: {'run_id': '7tm6txs7'},
                                3: {'run_id': 'dic1mu7c'},
                                4: {'run_id': '8ki0ym6o'}},
    # 'N $\\rightarrow$ DiffPool': {0: {'run_id': ''},
    #                    1: {'run_id': ''},
    #                    2: {'run_id': ''},
    #                    3: {'run_id': ''},
    #                    4: {'run_id': ''}},
}

DEVICE_RUN = 'cpu'


def print_metrics(model_name, runs_all):
    api = wandb.Api()
    metrics = {'acc': [], 'auc': [], 'sensitivity': [], 'specificity': []}

    for fold_num, run_info in runs_all.items():
        run_id = run_info['run_id']

        best_run = api.run(f'/tjiagom/st_extra/{run_id}')
        for metric in metrics.keys():
            metrics[metric].append(best_run.summary[f'values_test_{metric}'])

    print(model_name, end=' & ')
    print(f'{np.mean(metrics["auc"]):.2f} ({np.std(metrics["auc"]):.3f}) & '
          f'{np.mean(metrics["acc"]):.2f} ({np.std(metrics["acc"]):.3f}) & '
          f'{np.mean(metrics["sensitivity"]):.2f} ({np.std(metrics["sensitivity"]):.3f}) & '
          f'{np.mean(metrics["specificity"]):.2f} ({np.std(metrics["specificity"]):.3f})')


if __name__ == '__main__':
    for model_type, runs_all in best_runs_ukb.items():
        print_metrics(model_type, runs_all)
