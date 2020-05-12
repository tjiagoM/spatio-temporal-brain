import os
import random
from collections import deque
from sys import exit
from typing import Dict, Any

import numpy as np
import torch
import wandb
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader

from datasets import BrainDataset, HCPDataset, UKBDataset
from model import SpatioTemporalModel
from utils import create_name_for_brain_dataset, create_name_for_model, Normalisation, ConnType, ConvStrategy, \
    StratifiedGroupKFold, PoolingStrategy, AnalysisType, merge_y_and_others, EncodingStrategy, create_best_encoder_name, \
    SweepType, DatasetType, get_freer_gpu, free_gpu_info


import argparse
parser = argparse.ArgumentParser()
#parser.add_argument('--enc', help='enc help')
parser.add_argument('--thre', help='enc help')
args = parser.parse_args()

thre = int(args.thre)
#encoding_str = args.enc

print('Args are', thre)#, encoding_str)


name_dataset = create_name_for_brain_dataset(num_nodes=68,
                                                 time_length=1200,
                                                 target_var='gender',
                                                 threshold=thre,
                                                 normalisation=Normalisation('subject_norm'),
                                                 connectivity_type=ConnType('fmri'),
                                                 analysis_type=AnalysisType('st_unimodal'),
                                                 encoding_strategy=EncodingStrategy('none'),
                                                 dataset_type=DatasetType('hcp'),
                                             edge_weights=True)

print("Going for", name_dataset)
class_dataset = HCPDataset
dataset = class_dataset(root=name_dataset,
                        target_var='gender',
                        num_nodes=68,
                        threshold=thre,
                        connectivity_type=ConnType('fmri'),
                        normalisation=Normalisation('subject_norm'),
                        analysis_type=AnalysisType('st_unimodal'),
                        encoding_strategy=EncodingStrategy('none'),
                        time_length=1200,
                        edge_weights=True)
