###
###
## Put this in root folder to be able to properly run it
###

import argparse
from typing import Dict, Any

import numpy as np
import torch
import torch_geometric.utils as pyg_utils

from main_loop import generate_dataset, create_fold_generator
from utils import AnalysisType, DatasetType, ConnType, Normalisation, EncodingStrategy

# TODO: Make a table in README to compare my modifications to what cnslab.
#  eg. list of changes / table original<->changes. Highlight the only file I've edited (and remove others from repo)
#  eg: overall adj_matrix.npy is only on training (fold dependent, not overall)
#  ... train/val/test, whereas before it was only train/test therefore avoiding performance inflation.
#  ... change on model to include fold info (because of change on global adj matrix)
def save_cnslab_format(dataset_slice, fold_num: int, fold_name: str, dataset_type: str, analysis_type: str,
                       save_adj_mat=False):
    print(f'Saving CNSLAB format for {dataset_type}, fold_num={fold_num} and fold_name={fold_name}')

    labels = np.concatenate([data.y for data in dataset_slice])
    data = np.concatenate([data.x.reshape(1, 1, 490, 68, 1) for data in dataset_slice])

    if save_adj_mat:
        adj_mat = np.zeros((68, 68))
        for elem in dataset_slice:
            adj_mat += pyg_utils.to_dense_adj(edge_index=elem.edge_index, edge_attr=elem.edge_attr)[0, :, :, 0].numpy()
        adj_mat /= len(labels)

        # (68, 68) - diagonals with 1, and symmetric
        np.save(f'data/cnslab_{dataset_type}_{analysis_type}_adj_matrix_{fold_num}.npy', adj_mat)

    # (N, 1, 490, 68, 1)
    filename = f'data/cnslab_{dataset_type}_{analysis_type}_{fold_name}_data_{fold_num}.npy'
    np.save(filename, data)
    # (N,) ... array([0., 0., 1., 1., 0.])
    filename = f'data/cnslab_{dataset_type}_{analysis_type}_{fold_name}_label_{fold_num}.npy'
    np.save(filename, labels)


def run_for_specific_fold(fold_num: int, dataset_type: str, analysis_type: str):
    print(f'RUNNING FOR {fold_num} with {dataset_type}...')

    run_cfg: Dict[str, Any] = {
        'analysis_type': AnalysisType(analysis_type),
        'dataset_type': DatasetType(dataset_type),
        'num_nodes': 68,
        'param_conn_type': ConnType('fmri'), # Changed later
        'target_var': 'gender',
        'time_length': 490,
        'param_threshold': 10,
        'param_normalisation': Normalisation('subject_norm'),
        'param_encoding_strategy': EncodingStrategy('none'),
        'edge_weights': True,
        'split_to_test': fold_num,
        'multimodal_size': 0
    }

    if analysis_type == 'st_multimodal':
        run_cfg['param_conn_type'] = ConnType('struct')
        run_cfg['edge_weights'] = True

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

    save_cnslab_format(X_train_in, fold_num=fold_num, fold_name='train', save_adj_mat=True, dataset_type=dataset_type,
                       analysis_type=analysis_type)
    save_cnslab_format(X_val_in, fold_num=fold_num, fold_name='val', dataset_type=dataset_type,
                       analysis_type=analysis_type)
    save_cnslab_format(X_test_out, fold_num=fold_num, fold_name='test', dataset_type=dataset_type,
                       analysis_type=analysis_type)


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess data for SVM flatten baseline')
    parser.add_argument('--fold_num',
                        type=int,
                        choices=[1, 2, 3, 4, 5],
                        help='Fold number to process.')

    parser.add_argument('--dataset_type',
                        type=str,
                        choices=['ukb', 'hcp'],
                        help='Dataset to use.')

    parser.add_argument('--analysis_type',
                        type=str,
                        choices=['st_unimodal', 'st_multimodal'],
                        help='Which analysis type.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run_for_specific_fold(args.fold_num, args.dataset_type, args.analysis_type)
