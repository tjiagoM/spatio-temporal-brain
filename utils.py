import random
from collections import Counter, defaultdict
from enum import Enum, unique
from typing import NoReturn, Dict, Any

import fcntl
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBModel


@unique
# 3 first letters need to be different (for logging)
class SweepType(str, Enum):
    DIFFPOOL = 'diff_pool'
    NO_GNN = 'no_gnn'
    GCN = 'gcn'
    GAT = 'gat'
    META_NODE = 'node_meta'
    META_EDGE_NODE = 'edge_node_meta'
    FLATTEN_CORRS = 'flatten_corrs'


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
    NONE = 'none'


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
    ST_UNIMODAL_AVG = 'st_unimodal_avg'
    ST_MULTIMODAL = 'st_multimodal'
    ST_MULTIMODAL_AVG = 'st_multimodal_avg'
    FLATTEN_CORRS = 'flatten_corrs'
    FLATTEN_CORRS_THRESHOLD = 'flatten_corrs_threshold'


@unique
# 3 first letters need to be different (for logging)
class EncodingStrategy(str, Enum):
    NONE = 'none'
    AE3layers = '3layerAE'
    VAE3layers = '3layerVAE'
    STATS = 'stats'


def get_freer_gpu() -> int:
    """
    Considers that there is only GPU 0 and 1.
    :return:
    """
    # This option is not preventing when GPUs not being used yet
    # os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp_gpu')
    # memory_available = [int(x.split()[2]) for x in open('tmp_gpu', 'r').readlines()]
    # return np.argmax(memory_available)
    print('Overriding GPU info and getting GPU 0...')
    return 0
    print('Getting free GPU info...')
    gpu_to_use: int = 0
    with open('tmp_gpu.txt', 'r+') as fd:
        fcntl.flock(fd, fcntl.LOCK_EX)
        # Someone is using GPU 0 already
        info_file = fd.read()
        if info_file == 'server':
            print('Server usage, just give 0')
        elif info_file == '0':
            print('GPU 0 already in use')
            gpu_to_use = 1
        else:
            print('Reserving GPU 0')
            # Inform gpu 0 is now reserved
            fd.seek(0)
            fd.write('0')
            fd.truncate()
        fcntl.flock(fd, fcntl.LOCK_UN)

    return gpu_to_use


def free_gpu_info() -> NoReturn:
    print('Freeing GPU 0!')
    with open('tmp_gpu.txt', 'r+') as fd:
        fcntl.flock(fd, fcntl.LOCK_EX)
        info_file = fd.read()
        if info_file == 'server':
            print('Server usage, no need to free GPU')
        else:
            fd.seek(0)
            fd.write('')
            fd.truncate()

        fcntl.flock(fd, fcntl.LOCK_UN)


def merge_y_and_others(ys, indices):
    tmp = torch.cat([ys.long().view(-1, 1),
                     indices.view(-1, 1)], dim=1)
    return LabelEncoder().fit_transform([str(l) for l in tmp.numpy()])


def create_name_for_flattencorrs_dataset(run_cfg: Dict[str, Any]) -> str:
    prefix_location = './pytorch_data/unbalanced_'

    name_combination = '_'.join([run_cfg['dataset_type'].value,
                                 run_cfg['analysis_type'].value,
                                 run_cfg['param_conn_type'].value,
                                 str(run_cfg['num_nodes']),
                                 str(run_cfg['time_length'])
                                 ])

    return prefix_location + name_combination


def create_name_for_brain_dataset(num_nodes: int, time_length: int, target_var: str, threshold: int,
                                  connectivity_type: ConnType, normalisation: Normalisation,
                                  analysis_type: AnalysisType, dataset_type: DatasetType,
                                  encoding_strategy: EncodingStrategy, edge_weights: bool = False) -> str:
    if edge_weights:
        prefix_location = './pytorch_data/unbalanced_weights_'
    else:
        prefix_location = './pytorch_data/unbalanced_'

    name_combination = '_'.join(
        [target_var, dataset_type.value, analysis_type.value, encoding_strategy.value, connectivity_type.value,
         str(num_nodes), str(time_length), str(threshold), normalisation.value])

    return prefix_location + name_combination


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


def create_name_for_xgbmodel(run_cfg: Dict[str, Any], outer_split_num: int, model: XGBModel, inner_split_num: int,
                             prefix_location='logs/', suffix='.pkl') -> str:
    if run_cfg['analysis_type'] == AnalysisType.FLATTEN_CORRS:
        model_str_representation = run_cfg['analysis_type'].value
        for key in ['colsample_bylevel', 'colsample_bynode', 'colsample_bytree', 'gamma', 'learning_rate', 'max_depth',
                    'min_child_weight', 'n_estimators', 'subsample']:
            model_str_representation += key[-3:] + '_' + str(model.get_params()[key])
    return prefix_location + '_'.join([run_cfg['target_var'],
                                       run_cfg['dataset_type'].value,
                                       str(outer_split_num),
                                       str(inner_split_num),
                                       model_str_representation,
                                       str(run_cfg['num_nodes']),
                                       run_cfg['param_conn_type'].value
                                       ]) + suffix


def create_name_for_model(run_cfg: Dict[str, Any], model, outer_split_num: int, inner_split_num: int,
                          prefix_location='logs/', suffix='.pt') -> str:
    if run_cfg['analysis_type'] in [AnalysisType.ST_MULTIMODAL, AnalysisType.ST_UNIMODAL, AnalysisType.ST_UNIMODAL_AVG, AnalysisType.ST_MULTIMODAL_AVG]:
        model_str_representation = model.to_string_name()

    lr = round(run_cfg['param_lr'], 7)
    weight_decay = round(run_cfg['param_weight_decay'], 7)

    return prefix_location + '_'.join([run_cfg['target_var'],
                                       run_cfg['dataset_type'].value,
                                       str(outer_split_num),
                                       str(inner_split_num),
                                       model_str_representation,
                                       str(lr),
                                       str(weight_decay),
                                       str(run_cfg['param_threshold']),
                                       run_cfg['param_normalisation'].value[:3],
                                       str(run_cfg['num_nodes']),
                                       run_cfg['param_conn_type'].value
                                       ]) + suffix

def change_w_config_(w_config):
    '''
    Change w_config from wandb API to the one needed to the general functions in this project

    :param w_config:
    :return:
    '''
    w_config['analysis_type'] = AnalysisType(w_config['analysis_type'])
    w_config['dataset_type'] = DatasetType(w_config['dataset_type'])
    w_config['param_lr'] = w_config['lr']
    w_config['model_with_sigmoid'] = True
    w_config['param_activation'] = w_config['activation']
    w_config['param_channels_conv'] = w_config['channels_conv']
    w_config['param_conn_type'] = ConnType(w_config['conn_type'])
    w_config['param_conv_strategy'] = ConvStrategy(w_config['conv_strategy'])
    w_config['param_dropout'] = w_config['dropout']
    w_config['param_encoding_strategy'] = EncodingStrategy(w_config['encoding_strategy'])
    w_config['param_normalisation'] = Normalisation(w_config['normalisation'])
    w_config['param_num_gnn_layers'] = w_config['num_gnn_layers']
    w_config['param_pooling'] = PoolingStrategy(w_config['pooling'])
    w_config['param_weight_decay'] = w_config['weight_decay']

    w_config['sweep_type'] = SweepType(w_config['sweep_type'])
    w_config['param_gat_heads'] = 0
    if w_config['sweep_type'] == SweepType.GAT:
        w_config['param_gat_heads'] = w_config.gat_heads

    w_config['param_threshold'] = w_config['threshold']

    w_config['multimodal_size'] = 0
    #if w_config['analysis_type'] == AnalysisType.ST_MULTIMODAL:
    #    w_config['multimodal_size'] = 10
    #elif w_config['analysis_type'] == AnalysisType.ST_UNIMODAL:
    #    w_config['multimodal_size'] = 0

    if w_config['target_var'] in ['age', 'bmi']:
        w_config['model_with_sigmoid'] = False

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
