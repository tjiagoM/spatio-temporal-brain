import os
import random
from collections import Counter, defaultdict
from enum import Enum, unique
from typing import Union
from xgboost import XGBClassifier

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder


@unique
class SweepType(str, Enum):
    DIFFPOOL = 'diff_pool'
    NO_GNN = 'no_gnn'
    GCN = 'gcn'
    GAT = 'gat'


@unique
# 3 first letters need to be different (for logging)
class Normalisation(str, Enum):
    NONE = 'no_norm'
    ROI = 'roi_norm'
    SUBJECT = 'subject_norm'


@unique
class DatasetType(str, Enum):
    HCP = 'hcp'
    UKB = 'ukb'


@unique
class ConnType(str, Enum):
    FMRI = 'fmri'
    STRUCT = 'struct'


@unique
# 3 first letters need to be different (for logging)
class ConvStrategy(str, Enum):
    CNN_ENTIRE = 'entire'
    TCN_ENTIRE = 'tcn_entire'


@unique
# 3 first letters need to be different (for logging)
class PoolingStrategy(str, Enum):
    MEAN = 'mean'
    DIFFPOOL = 'diff_pool'
    CONCAT = 'concat'


@unique
class AnalysisType(str, Enum):
    """
    ST_* represent the type of data in each node
    FLATTEN_* represents the xgboost baseline
    """
    ST_UNIMODAL = 'st_unimodal'
    ST_MULTIMODAL = 'st_multimodal'
    FLATTEN_CORRS = 'flatten_corrs'
    FLATTEN_CORRS_THRESHOLD = 'flatten_corrs_threshold'


@unique
class EncodingStrategy(str, Enum):
    NONE = 'none'
    AE3layers = '3layerAE'
    VAE3layers = '3layerVAE'


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


def merge_y_and_others(ys, indices):
    tmp = torch.cat([ys.long().view(-1, 1),
                     indices.view(-1, 1)], dim=1)
    return LabelEncoder().fit_transform([str(l) for l in tmp.numpy()])


def create_name_for_brain_dataset(num_nodes: int, time_length: int, target_var: str, threshold: int,
                                  connectivity_type: ConnType, normalisation: Normalisation,
                                  analysis_type: AnalysisType, dataset_type: DatasetType):
    prefix_location = './pytorch_data/unbalanced_'

    name_combination = '_'.join(
        [target_var, dataset_type.value, analysis_type.value, connectivity_type.value, str(num_nodes), str(time_length),
         str(threshold), normalisation.value])

    return prefix_location + name_combination


def get_best_model_paths(analysis_type, num_nodes, time_length, target_var,
                         fold_num, conn_type, num_epochs,
                         sweep_type,
                         first_time=False,
                         prefix_location='logs/'):
    m_name = '_'.join([analysis_type, str(num_nodes), str(time_length), target_var,
                       str(fold_num), conn_type, sweep_type, str(num_epochs)])

    loss_val = prefix_location + m_name + '_loss.npy'
    model_name = prefix_location + m_name + '_name.txt'

    if (not os.path.exists(loss_val)) or first_time:
        np.save(file=loss_val, arr=np.array([9999.99], dtype=float))
        with open(model_name, 'w') as f:
            f.write('None')

    return loss_val, model_name


def create_best_encoder_name(ts_length, outer_split_num, encoder_name,
                             prefix_location='logs/',
                             suffix='.pth'):
    return f'{prefix_location}{encoder_name}_{ts_length}_{outer_split_num}_best{suffix}'


def create_name_for_encoder_model(ts_length, outer_split_num, encoder_name,
                                  params,
                                  prefix_location='logs/',
                                  suffix='.pth'):
    return prefix_location + '_'.join([encoder_name,
                                       str(ts_length),
                                       str(outer_split_num),
                                       str(params['weight_decay']),
                                       str(params['lr'])
                                       ]) + suffix


def create_name_for_model(target_var: str, model, outer_split_num: int,
                          inner_split_num: int, n_epochs: int, threshold: int, batch_size: int, num_nodes: int,
                          conn_type: ConnType, normalisation: Normalisation, analysis_type: AnalysisType,
                          metric_evaluated: str, dataset_type: DatasetType,
                          lr=None, weight_decay=None,
                          prefix_location='logs/',
                          suffix='.pth') -> str:
    if analysis_type in [AnalysisType.ST_MULTIMODAL, AnalysisType.ST_UNIMODAL]:
        model_str_representation = model.to_string_name()
    elif analysis_type == AnalysisType.FLATTEN_CORRS or analysis_type == AnalysisType.FLATTEN_CORRS_THRESHOLD:
        suffix = '.pkl'
        model_str_representation = analysis_type.value
        for key in ['min_child_weight', 'gamma', 'subsample', 'colsample_bytree', 'max_depth', 'n_estimators']:
            model_str_representation += key[:3] + '_' + str(model.get_xgb_params()[key])

    return prefix_location + '_'.join([target_var,
                                       dataset_type.value,
                                       str(outer_split_num),
                                       str(inner_split_num),
                                       metric_evaluated,
                                       model_str_representation,
                                       str(lr),
                                       str(weight_decay),
                                       str(n_epochs),
                                       'TH_' + str(threshold),
                                       normalisation.value[:3],
                                       str(batch_size),
                                       str(num_nodes),
                                       conn_type.value
                                       ]) + suffix


# From https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
class StratifiedGroupKFold:

    def __init__(self, n_splits=5, random_state=0):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X, y, groups):
        labels_num = np.max(y) + 1
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
        y_distr = Counter()
        for label, g in zip(y, groups):
            y_counts_per_group[g][label] += 1
            y_distr[label] += 1

        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
        groups_per_fold = defaultdict(set)

        def eval_y_counts_per_fold(y_counts, fold):
            y_counts_per_fold[fold] += y_counts
            std_per_label = []
            for label in range(labels_num):
                label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(self.n_splits)])
                std_per_label.append(label_std)
            y_counts_per_fold[fold] -= y_counts
            return np.mean(std_per_label)

        groups_and_y_counts = list(y_counts_per_group.items())
        random.Random(self.random_state).shuffle(groups_and_y_counts)

        for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for i in range(self.n_splits):
                fold_eval = eval_y_counts_per_fold(y_counts, i)
                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = i
            y_counts_per_fold[best_fold] += y_counts
            groups_per_fold[best_fold].add(g)

        all_groups = set(groups)
        for i in range(self.n_splits):
            train_groups = all_groups - groups_per_fold[i]
            test_groups = groups_per_fold[i]

            train_indices = [i for i, g in enumerate(groups) if g in train_groups]
            test_indices = [i for i, g in enumerate(groups) if g in test_groups]

            yield train_indices, test_indices
