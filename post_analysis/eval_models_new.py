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
    'N + E $\\rightarrow$ Concat': {0: {'run_id': 'fbaff478'},
                                    1: {'run_id': 'i0ii76x1'},
                                    2: {'run_id': 'pm45qx39'},
                                    3: {'run_id': '1579q23v'},
                                    4: {'run_id': '3iaecgum'}},
     'N + E $\\rightarrow$ DiffPool': {0: {'run_id': 'w240uvas'},
                        1: {'run_id': 'zeyzk9cc'},
                        2: {'run_id': '99oyqkv7'},
                        3: {'run_id': '4uha08sz'},
                        4: {'run_id': 'p0qo4sru'}},
    'N $\\rightarrow$ Concat': {0: {'run_id': '5ra9icg7'},
                                1: {'run_id': 'f2n8s6gw'},
                                2: {'run_id': '3vunou40'},
                                3: {'run_id': 'q6dl8u6b'},
                                4: {'run_id': '90e7orj6'}},
     'N $\\rightarrow$ DiffPool': {0: {'run_id': '1l6q4muy'},
                        1: {'run_id': 'y1lyxyvm'},
                        2: {'run_id': 'xahinw23'},
                        3: {'run_id': 'lnfvnnd9'},
                        4: {'run_id': 'fvw8ordx'}},
    # ' $\\rightarrow$ DiffPool': {0: {'run_id': ''},
    #                    1: {'run_id': ''},
    #                    2: {'run_id': ''},
    #                    3: {'run_id': ''},
    #                    4: {'run_id': ''}},
     ' $\\rightarrow$ Concat': {0: {'run_id': 'jfi7tuxz'},
                        1: {'run_id': 'qd0gbu2y'},
                        2: {'run_id': 'k8tfwp74'},
                        3: {'run_id': 'rmgvjwqe'},
                        4: {'run_id': 'agqu30gc'}},
}

best_runs_hcp_fmri = {
     'N + E $\\rightarrow$ Concat': {0: {'run_id': 'tdztbchl'},
                       1: {'run_id': '5t0e0oel'},
                       2: {'run_id': '6d5qb8c7'},
                       3: {'run_id': 'pw7h3kub'},
                       4: {'run_id': '4txocl2p'}},
     'N + E $\\rightarrow$ DiffPool': {0: {'run_id': '3oo814km'},
                        1: {'run_id': 'onvh74qg'},
                        2: {'run_id': 'ybvfbb4b'},
                        3: {'run_id': 'azn8u5ud'},
                        4: {'run_id': 'wjjenvbp'}},
    'N $\\rightarrow$ Concat': {0: {'run_id': 'hovygim0'},
                                1: {'run_id': '3cej2yzp'},
                                2: {'run_id': 'gmfaxo2e'},
                                3: {'run_id': 'fv4xi5ve'},
                                4: {'run_id': 'c5hvjkq8'}},
     'N $\\rightarrow$ DiffPool': {0: {'run_id': '4b3wepsy'},
                        1: {'run_id': 'ub1r8snd'},
                        2: {'run_id': 'fzfu7fid'},
                        3: {'run_id': 'si0n1zto'},
                        4: {'run_id': '4l00ghh5'}},
    'N $\\rightarrow$ DiffPool (Mean)': {0: {'run_id': 'ixncz5xu'},
                        1: {'run_id': '5j5o8euj'},
                        2: {'run_id': 'lypdotyy'},
                        3: {'run_id': 'jsnbufgc'},
                        4: {'run_id': '82ircjmb'}},
    ' $\\rightarrow$ Concat ': {0: {'run_id': 'p0neg8hu'},
                        1: {'run_id': 'zs0yym6c'},
                        2: {'run_id': 'b7bjrc8k'},
                        3: {'run_id': 'ls0u853s'},
                        4: {'run_id': 'in6corcd'}},
    ' $\\rightarrow$ DiffPool (Sum) ': {0: {'run_id': 'm4542j80'},
                        1: {'run_id': 'd16h4763'},
                        2: {'run_id': 'dz3alt0l'},
                        3: {'run_id': 'cup9o1c7'},
                        4: {'run_id': 'j3mjd4xj'}}
}

best_runs_hcp_struct = {
    'N + E $\\rightarrow$ Concat': {0: {'run_id': 'wrh9gt23'},
                                    1: {'run_id': 'tnnogscw'},
                                    2: {'run_id': 'eebx30ts'},
                                    3: {'run_id': 'bk2ql2m9'},
                                    4: {'run_id': '8atwfm0e'}},
     'N + E $\\rightarrow$ DiffPool': {0: {'run_id': '2c0oqarl'},
                        1: {'run_id': '7rzqoqnx'},
                        2: {'run_id': 'rxfa5aq6'},
                        3: {'run_id': 'ndy44vc1'},
                        4: {'run_id': 'arawexn0'}},
    'N $\\rightarrow$ Concat': {0: {'run_id': 'xfjrfwmv'},
                                1: {'run_id': 'luzbl8k9'},
                                2: {'run_id': '7tm6txs7'},
                                3: {'run_id': 'dic1mu7c'},
                                4: {'run_id': '8ki0ym6o'}},
     'N $\\rightarrow$ DiffPool': {0: {'run_id': 'o1fdm77q'},
                        1: {'run_id': 'l1hyry51'},
                        2: {'run_id': 't31f7pjz'},
                        3: {'run_id': 'iiqatrvc'},
                        4: {'run_id': 'n9i2w0rs'}},
    'N $\\rightarrow$ DiffPool (Mean)': {0: {'run_id': '5geocaib'},
                        1: {'run_id': '4ijultjc'},
                        2: {'run_id': 'zff6mg3i'},
                        3: {'run_id': 'em4pfn7r'},
                        4: {'run_id': 'xt71mwep'}},
     ' $\\rightarrow$ DiffPool (Sum)': {0: {'run_id': 'orf1xt53'},
                        1: {'run_id': 'gm0zubbs'},
                        2: {'run_id': '6drdrvbz'},
                        3: {'run_id': 'zto6olyj'},
                        4: {'run_id': 'ze09wvbz'}}
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
    for model_type, runs_all in best_runs_hcp_struct.items():
        print_metrics(model_type, runs_all)
    print('----')
    for model_type, runs_all in best_runs_hcp_fmri.items():
        print_metrics(model_type, runs_all)
