import argparse
from typing import Dict, Any

import numpy as np
import torch
from scipy import stats

from main_loop import generate_dataset, create_fold_generator
from utils import AnalysisType, DatasetType, ConnType, Normalisation, EncodingStrategy


# TODO: Make a table in README to compare my modifications to what cnslab.
#  eg. list of changes / table original<->changes. Highlight the only file I've edited (and remove others from repo)
#  eg: overall adj_matrix.npy is only on training (fold dependent, not overall)
#  ... train/val/test, whereas before it was only train/test therefore avoiding performance inflation.
#  ... change on model to include fold info (because of change on global adj matrix)
def save_cnslab_format(dataset_slice, fold_num: int, fold_name: str, save_adj_mat=False):
    print(f'Saving CNSLAB format for fold_num={fold_num} and fold_name={fold_name}')
    L = 490
    N_ROI = 68

    ys = np.concatenate([data.y for data in dataset_slice])
    data = np.zeros((len(dataset_slice), 1, L, N_ROI, 1))
    label = np.zeros((ys.shape[0],))

    # load all data
    idx = 0
    data_all = None

    # for i in range(demo.shape[0]):
    for data_elem in dataset_slice:
        # subject_string = format(int(demo[i, 0]), '06d')
        # print(subject_string)
        # filename_full = 'data/hcp_tc_npy_22/' + subject_string + '_cortex.npy'
        # full_sequence = np.load(filename_full)
        full_sequence = data_elem.x.numpy()

        # if full_sequence.shape[1] < S + L:
        #    continue

        # full_sequence = full_sequence[:, S:S + L];
        z_sequence = stats.zscore(full_sequence, axis=1)

        if len(z_sequence[np.isnan(z_sequence)]) > 0:
            z_sequence[np.isnan(z_sequence)] = 0

        if data_all is None:
            data_all = z_sequence
        else:
            data_all = np.concatenate((data_all, z_sequence), axis=1)

        data[idx, 0, :, :, 0] = np.transpose(z_sequence)
        label[idx] = data_elem.y.item()  # demo[i, 1]
        idx = idx + 1

    if save_adj_mat:
        # compute adj matrix
        A = np.zeros((N_ROI, N_ROI))
        for i in range(N_ROI):
            for j in range(i, N_ROI):
                if i == j:
                    A[i][j] = 1
                else:
                    A[i][j] = abs(np.corrcoef(data_all[i, :], data_all[j, :])[0][1])  # get value from corrcoef matrix
                    A[j][i] = A[i][j]

        np.save(f'data/cnslab_adj_matrix_{fold_num}.npy', A)

    filename = f'data/cnslab_{fold_name}_data_{fold_num}.npy'
    np.save(filename, data)
    filename = f'data/cnslab_{fold_name}_label_{fold_num}.npy'
    np.save(filename, label)


def run_for_specific_fold(fold_num: int):
    print(f'RUNNING FOR {fold_num}...')

    run_cfg: Dict[str, Any] = {
        'analysis_type': AnalysisType('st_unimodal'),
        'dataset_type': DatasetType('ukb'),
        'num_nodes': 68,
        'param_conn_type': ConnType('fmri'),
        'target_var': 'gender',
        'time_length': 490,
        'param_threshold': 5,  # Doesn't matter, only looking for timeseries?
        'param_normalisation': Normalisation('no_norm'),  # TODO: check this will go ok for normalisation
        'param_encoding_strategy': EncodingStrategy('none'),
        'edge_weights': False,
        'split_to_test': fold_num
    }

    N_OUT_SPLITS: int = 5
    N_INNER_SPLITS: int = 5

    dataset = generate_dataset(run_cfg)
    skf_outer_generator = create_fold_generator(dataset, run_cfg, N_OUT_SPLITS)

    # Getting train / test folds
    outer_split_num: int = 0
    for train_index, test_index in skf_outer_generator:
        outer_split_num += 1
        # Only run for the specific fold defined in the script arguments.
        if outer_split_num != run_cfg['split_to_test']:
            continue

        X_train_out = dataset[torch.tensor(train_index)]
        X_test_out = dataset[torch.tensor(test_index)]

        break

    # Train / test sets defined, running the rest
    print('Positive classes:', sum([data.y.item() for data in X_train_out]),
          '/', sum([data.y.item() for data in X_test_out]))

    skf_inner_generator = create_fold_generator(X_train_out, run_cfg, N_INNER_SPLITS)

    #################
    # Main inner-loop
    #################
    inner_loop_run: int = 0
    for inner_train_index, inner_val_index in skf_inner_generator:
        inner_loop_run += 1

        X_train_in = X_train_out[torch.tensor(inner_train_index)]
        X_val_in = X_train_out[torch.tensor(inner_val_index)]
        print("Inner Size is:", len(X_train_in), "/", len(X_val_in))
        print("Inner Positive classes:", sum([data.y.item() for data in X_train_in]),
              "/", sum([data.y.item() for data in X_val_in]))

        # One inner loop only
        break

    save_cnslab_format(X_train_in, fold_num=fold_num, fold_name='train', save_adj_mat=True)
    save_cnslab_format(X_val_in, fold_num=fold_num, fold_name='val')
    save_cnslab_format(X_test_out, fold_num=fold_num, fold_name='test')


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess UKBiobank data for CNSLAB baseline')
    parser.add_argument('--fold_num',
                        type=int,
                        choices=[1, 2, 3, 4, 5],
                        help='Fold number to process.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run_for_specific_fold(args.fold_num)