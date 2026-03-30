import numpy as np
import wandb

best_runs_tmp = {
    'tmp': {0: {'run_id': ''},
            1: {'run_id': ''},
            2: {'run_id': ''},
            3: {'run_id': ''},
            4: {'run_id': ''}},
}

best_runs_notcn = {
    'N $\\rightarrow$ Concat': {0: {'run_id': 'twlgqhys'},
            1: {'run_id': 'dai5zyud'},
            2: {'run_id': 't84plo2z'},
            3: {'run_id': 'onpj78rz'},
            4: {'run_id': '5vv10ln1'}},
    'N + E $\\rightarrow$ Concat': {0: {'run_id': 's3nom54e'},
            1: {'run_id': '35kpz3r7'},
            2: {'run_id': 'gftjqol6'},
            3: {'run_id': 'cz5f4mdy'},
            4: {'run_id': 'd8k7aqlp'}}
}

best_runs_lstm = {
    'N $\\rightarrow$ Concat': {0: {'run_id': '3dimo8ds'},
            1: {'run_id': 'z55l7ytm'},
            2: {'run_id': '5023mak8'},
            3: {'run_id': 'sa8x2q1q'},
            4: {'run_id': 'y1bpo800'}},
    ' $\\rightarrow$ Concat': {0: {'run_id': 'ho395tt8'},
            1: {'run_id': 'xxk9nurr'},
            2: {'run_id': 'x32oa9ai'},
            3: {'run_id': 'w4uqg2xs'},
            4: {'run_id': '9czzokj5'}}
}

best_runs_lstm = {
    'N + E $\\rightarrow$ Concat': {0: {'run_id': 'ljy13d2u'},
                                    1: {'run_id': 'dfrjiq6h'},
                                    2: {'run_id': 'd64hmq3r'},
                                    3: {'run_id': 'p6zk1rky'},
                                    4: {'run_id': 'ou9ankw2'}},
     'N + E $\\rightarrow$ DiffPool': {0: {'run_id': 'ammw21dz'},
                        1: {'run_id': 'epksptq6'},
                        2: {'run_id': 'xq82baf2'},
                        3: {'run_id': 'c4qq3i85'},
                        4: {'run_id': 'b5csi1i8'}},
    'N $\\rightarrow$ Concat': {0: {'run_id': '1m3p843n'},
                                1: {'run_id': 'wj7o5foo'},
                                2: {'run_id': 't6n9mcku'},
                                3: {'run_id': 'ibxt1b2f'},
                                4: {'run_id': 'ebihuz6a'}},
     'N $\\rightarrow$ DiffPool': {0: {'run_id': 'ykde2757'},
                        1: {'run_id': 'puzzjfbm'},
                        2: {'run_id': '1b15mou6'},
                        3: {'run_id': 'mg0sc7x1'},
                        4: {'run_id': 'w9hhamx6'}},
     ' $\\rightarrow$ DiffPool': {0: {'run_id': 'gwdi0jt8'},
                        1: {'run_id': '8rra5tpt'},
                        2: {'run_id': '8mdhktq5'},
                        3: {'run_id': 'ozt90m9b'},
                        4: {'run_id': 'fj8cqv9l'}},
     ' $\\rightarrow$ Concat': {0: {'run_id': '6w6cobrm'},
                        1: {'run_id': 'd73nu5pi'},
                        2: {'run_id': '7pmdp443'},
                        3: {'run_id': 'zwonnsin'},
                        4: {'run_id': 'gicm95oe'}},
}

best_runs_ukb = {
    'N + E $\\rightarrow$ Concat': {0: {'run_id': 'fbaff478'},
                                    1: {'run_id': 'i0ii76x1'},
                                    2: {'run_id': 'pm45qx39'},
                                    3: {'run_id': '1579q23v'},
                                    4: {'run_id': '3iaecgum'}},
     'N + E $\\rightarrow$ DiffPool': {0: {'run_id': 'rc4mpxrz'},
                        1: {'run_id': 'rg1dgv3c'},
                        2: {'run_id': 'otv4x0ym'},
                        3: {'run_id': '5kiz6m7k'},
                        4: {'run_id': 'qxhwwpp4'}},
    'N $\\rightarrow$ Concat': {0: {'run_id': '5ra9icg7'},
                                1: {'run_id': 'f2n8s6gw'},
                                2: {'run_id': '3vunou40'},
                                3: {'run_id': 'q6dl8u6b'},
                                4: {'run_id': '90e7orj6'}},
     'N $\\rightarrow$ DiffPool': {0: {'run_id': '7lv9g3zl'},
                        1: {'run_id': 'ehwgffad'},
                        2: {'run_id': 'zsv4i7ln'},
                        3: {'run_id': '6of50lbk'},
                        4: {'run_id': '7b3vfv7u'}},
     ' $\\rightarrow$ DiffPool': {0: {'run_id': 'apzymi22'},
                        1: {'run_id': 'bc8t4ulw'},
                        2: {'run_id': '4zpxljpq'},
                        3: {'run_id': 'tttzzu2m'},
                        4: {'run_id': 'qwnipyru'}},
     ' $\\rightarrow$ Concat': {0: {'run_id': 'jfi7tuxz'},
                        1: {'run_id': 'qd0gbu2y'},
                        2: {'run_id': 'k8tfwp74'},
                        3: {'run_id': 'rmgvjwqe'},
                        4: {'run_id': 'agqu30gc'}},
}

best_runs_thre100_ukb = {
    'N + E $\\rightarrow$ Concat': {0: {'run_id': 'ieh2jbce'},
                                    1: {'run_id': '0lsnau4z'},
                                    2: {'run_id': 'k32c1ckp'},
                                    3: {'run_id': 'gqv4vfoo'},
                                    4: {'run_id': 'ql7jjb9b'}},
     'N + E $\\rightarrow$ DiffPool': {0: {'run_id': 'wb93e183'},
                        1: {'run_id': '15jmbs0x'},
                        2: {'run_id': 'tr3362d3'},
                        3: {'run_id': '6gbh7jt1'},
                        4: {'run_id': '96wkdl8i'}},
    'N $\\rightarrow$ Concat': {0: {'run_id': '06ybqc3d'},
                                1: {'run_id': 'je48s665'},
                                2: {'run_id': 'w632q3du'},
                                3: {'run_id': 'z39xsinv'},
                                4: {'run_id': '650d93ga'}},
     'N $\\rightarrow$ DiffPool': {0: {'run_id': 'v0nljvcf'},
                        1: {'run_id': 'dncxffke'},
                        2: {'run_id': 'dvc6767t'},
                        3: {'run_id': 'vliojzuh'},
                        4: {'run_id': 's1cjijtu'}}
}

best_runs_hcp_fmri = {
     'N + E $\\rightarrow$ Concat': {0: {'run_id': 'tdztbchl'},
                       1: {'run_id': '5t0e0oel'},
                       2: {'run_id': '6d5qb8c7'},
                       3: {'run_id': 'pw7h3kub'},
                       4: {'run_id': '4txocl2p'}},
     'N + E $\\rightarrow$ DiffPool': {0: {'run_id': 'rdzluequ'},
                        1: {'run_id': 'gawme2wn'},
                        2: {'run_id': 'k4dikjin'},
                        3: {'run_id': '444xtahu'},
                        4: {'run_id': 'fcvpo0eg'}},
    'N $\\rightarrow$ Concat': {0: {'run_id': 'hovygim0'},
                                1: {'run_id': '3cej2yzp'},
                                2: {'run_id': 'gmfaxo2e'},
                                3: {'run_id': 'fv4xi5ve'},
                                4: {'run_id': 'c5hvjkq8'}},
     'N $\\rightarrow$ DiffPool': {0: {'run_id': 'hso8zyod'},
                        1: {'run_id': 'y3jly9zt'},
                        2: {'run_id': '54h5kfpo'},
                        3: {'run_id': 'ys57gt39'},
                        4: {'run_id': 'ga4x9bem'}},
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
     'N + E $\\rightarrow$ DiffPool': {0: {'run_id': 'uepre0y4'},
                        1: {'run_id': 'itz6bldz'},
                        2: {'run_id': 'b9ieznpd'},
                        3: {'run_id': 'jmal2ghv'},
                        4: {'run_id': 'lf56w15e'}},
    'N $\\rightarrow$ Concat': {0: {'run_id': 'xfjrfwmv'},
                                1: {'run_id': 'luzbl8k9'},
                                2: {'run_id': '7tm6txs7'},
                                3: {'run_id': 'dic1mu7c'},
                                4: {'run_id': '8ki0ym6o'}},
     'N $\\rightarrow$ DiffPool': {0: {'run_id': 'pv0zqdkd'},
                        1: {'run_id': 'ljwobaut'},
                        2: {'run_id': '3om34gbi'},
                        3: {'run_id': 'a9i1spyz'},
                        4: {'run_id': '5a5dolm3'}},
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
    #for model_type, runs_all in best_runs_ukb.items():
    #    print_metrics(model_type, runs_all)
    print('----')
    for model_type, runs_all in best_runs_notcn.items():
        print_metrics(model_type, runs_all)
