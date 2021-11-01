import os
import pickle
import random
from collections import deque
from sys import exit
from typing import Dict, Any, Union

import numpy as np
import pandas as pd
import torch
import wandb
from scipy.stats import stats
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch_geometric.data import DataLoader
from xgboost import XGBClassifier, XGBRegressor, XGBModel

from datasets import BrainDataset, HCPDataset, UKBDataset, FlattenCorrsDataset
from model import SpatioTemporalModel
from utils import create_name_for_brain_dataset, create_name_for_model, Normalisation, ConnType, ConvStrategy, \
    StratifiedGroupKFold, PoolingStrategy, AnalysisType, merge_y_and_others, EncodingStrategy, create_best_encoder_name, \
    SweepType, DatasetType, get_freer_gpu, free_gpu_info, create_name_for_flattencorrs_dataset, \
    create_name_for_xgbmodel, LRScheduler, Optimiser, EarlyStopping, ModelEmaV2, calculate_indegree_histogram

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class MSLELoss(torch.nn.Module):
    def __init__(self):
        super(MSLELoss, self).__init__()
        self.squared_error = torch.nn.MSELoss(reduction='none')

    def forward(self, y_hat, y):
        # the log(predictions) corresponding to no data should be set to 0
        log_y_hat = y_hat.log()  # where(y_hat > 0, torch.zeros_like(y_hat)).log()
        # the we set the log(labels) that correspond to no data to be 0 as well
        log_y = y.log()  # where(y > 0, torch.zeros_like(y)).log()
        # where there is no data log_y_hat = log_y = 0, so the squared error will be 0 in these places
        loss = self.squared_error(log_y_hat, log_y)
        # print('A', loss.shape)
        # print(loss.shape, loss)
        # loss = torch.sum(loss, dim=1)
        # if not sum_losses:
        #    loss = loss / seq_length.clamp(min=1)
        return loss.mean()


def train_model(model, train_loader, optimizer, run_cfg, model_ema=None, label_scaler=None):
    pooling_mechanism = run_cfg['param_pooling']
    device = run_cfg['device_run']

    model.train()
    loss_all = 0
    loss_all_link = 0
    loss_all_ent = 0
    if label_scaler is None:
        criterion = torch.nn.BCELoss()
    else:
        criterion = torch.nn.SmoothL1Loss()

    grads = []
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        if pooling_mechanism in [PoolingStrategy.DIFFPOOL, PoolingStrategy.DP_MAX, PoolingStrategy.DP_ADD, PoolingStrategy.DP_MEAN]:
            output_batch, link_loss, ent_loss = model(data)
            output_batch = output_batch.clamp(0, 1)  # For NaNs
            output_batch = torch.where(torch.isnan(output_batch), torch.zeros_like(output_batch), output_batch) # For NaNs
            loss = criterion(output_batch, data.y.unsqueeze(1)) + link_loss + ent_loss
            loss_b_link = link_loss
            loss_b_ent = ent_loss
        else:
            output_batch = model(data)
            output_batch = output_batch.clamp(0, 1) # For NaNs
            output_batch = torch.where(torch.isnan(output_batch), torch.zeros_like(output_batch), output_batch) # For NaNs
            loss = criterion(output_batch, data.y.unsqueeze(1))

        loss.backward()

        if run_cfg['final_mlp_layers'] == 1:
            grads.extend(model.final_linear.weight.grad.flatten().cpu().tolist())
        else:
            grads.extend(model.final_linear[-1].weight.grad.flatten().cpu().tolist())

        loss_all += loss.item() * data.num_graphs
        if pooling_mechanism in [PoolingStrategy.DIFFPOOL, PoolingStrategy.DP_MAX, PoolingStrategy.DP_ADD, PoolingStrategy.DP_MEAN]:
            loss_all_link += loss_b_link.item() * data.num_graphs
            loss_all_ent += loss_b_ent.item() * data.num_graphs

        torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        optimizer.step()
        if model_ema is not None:
            model_ema.update(model)
    print("GRAD", np.mean(grads), np.max(grads), np.std(grads))
    # len(train_loader) gives the number of batches
    # len(train_loader.dataset) gives the number of graphs

    # Returning a weighted average according to number of graphs
    return loss_all / len(train_loader.dataset), loss_all_link / len(train_loader.dataset), loss_all_ent / len(
        train_loader.dataset)


def return_regressor_metrics(labels, pred_prob, label_scaler=None, loss_value=None, link_loss_value=None,
                             ent_loss_value=None):
    if label_scaler is not None:
        labels = label_scaler.inverse_transform(labels.reshape(-1, 1))[:, 0]
        pred_prob = label_scaler.inverse_transform(pred_prob.reshape(-1, 1))[:, 0]

    print('First 5 values:', labels.shape, labels[:5], pred_prob.shape, pred_prob[:5])
    r2 = r2_score(labels, pred_prob)
    r = stats.pearsonr(labels, pred_prob)[0]

    return {'loss': loss_value,
            'link_loss': link_loss_value,
            'ent_loss': ent_loss_value,
            'r2': r2,
            'r': r
            }


def return_classifier_metrics(labels, pred_binary, pred_prob, loss_value=None, link_loss_value=None,
                              ent_loss_value=None,
                              flatten_approach: bool = False):
    roc_auc = roc_auc_score(labels, pred_prob)
    acc = accuracy_score(labels, pred_binary)
    f1 = f1_score(labels, pred_binary, zero_division=0)
    report = classification_report(labels, pred_binary, output_dict=True, zero_division=0)

    if not flatten_approach:
        sens = report['1.0']['recall']
        spec = report['0.0']['recall']
    else:
        sens = report['1']['recall']
        spec = report['0']['recall']

    return {'loss': loss_value,
            'link_loss': link_loss_value,
            'ent_loss': ent_loss_value,
            'auc': roc_auc,
            'acc': acc,
            'f1': f1,
            'sensitivity': sens,
            'specificity': spec
            }


def evaluate_model(model, loader, run_cfg, label_scaler=None):
    pooling_mechanism = run_cfg['param_pooling']
    device = run_cfg['device_run']

    model.eval()
    if label_scaler is None:
        criterion = torch.nn.BCELoss()
    else:
        criterion = torch.nn.SmoothL1Loss()

    predictions = []
    labels = []
    test_error = 0
    test_link_loss = 0
    test_ent_loss = 0

    for data in loader:
        with torch.no_grad():
            data = data.to(device)
            if pooling_mechanism in [PoolingStrategy.DIFFPOOL, PoolingStrategy.DP_MAX, PoolingStrategy.DP_ADD, PoolingStrategy.DP_MEAN]:
                output_batch, link_loss, ent_loss = model(data)
                output_batch = output_batch.clamp(0, 1)  # For NaNs
                output_batch = torch.where(torch.isnan(output_batch), torch.zeros_like(output_batch), output_batch)  # For NaNs
                # output_batch = output_batch.flatten()
                loss = criterion(output_batch, data.y.unsqueeze(1)) + link_loss + ent_loss
                loss_b_link = link_loss
                loss_b_ent = ent_loss
            else:
                output_batch = model(data)
                output_batch = output_batch.clamp(0, 1)  # For NaNs
                output_batch = torch.where(torch.isnan(output_batch), torch.zeros_like(output_batch), output_batch)  # For NaNs
                # output_batch = output_batch.flatten()
                loss = criterion(output_batch, data.y.unsqueeze(1))

            test_error += loss.item() * data.num_graphs
            if pooling_mechanism in [PoolingStrategy.DIFFPOOL, PoolingStrategy.DP_MAX, PoolingStrategy.DP_ADD, PoolingStrategy.DP_MEAN]:
                test_link_loss += loss_b_link.item() * data.num_graphs
                test_ent_loss += loss_b_ent.item() * data.num_graphs

            pred = output_batch.flatten().detach().cpu().numpy()

            label = data.y.detach().cpu().numpy()
            predictions.append(pred)
            labels.append(label)
    predictions = np.hstack(predictions)
    labels = np.hstack(labels)

    if label_scaler is None:
        pred_binary = np.where(predictions > 0.5, 1, 0)
        return return_classifier_metrics(labels, pred_binary, predictions,
                                         loss_value=test_error / len(loader.dataset),
                                         link_loss_value=test_link_loss / len(loader.dataset),
                                         ent_loss_value=test_ent_loss / len(loader.dataset))
    else:
        return return_regressor_metrics(labels, predictions,
                                        label_scaler=label_scaler,
                                        loss_value=test_error / len(loader.dataset),
                                        link_loss_value=test_link_loss / len(loader.dataset),
                                        ent_loss_value=test_ent_loss / len(loader.dataset))


def training_step(outer_split_no, inner_split_no, epoch, model, train_loader, val_loader, optimizer,
                  model_ema, run_cfg, label_scaler=None):
    loss, link_loss, ent_loss = train_model(model, train_loader, optimizer, run_cfg=run_cfg,
                                            model_ema=model_ema, label_scaler=label_scaler)
    train_metrics = evaluate_model(model, train_loader, run_cfg=run_cfg, label_scaler=label_scaler)
    if model_ema is not None:
        val_metrics = evaluate_model(model_ema.module, val_loader, run_cfg=run_cfg, label_scaler=label_scaler)
    else:
        val_metrics = evaluate_model(model, val_loader, run_cfg=run_cfg, label_scaler=label_scaler)

    if label_scaler is None:
        print(
            '{:1d}-{:1d}-Epoch: {:03d}, Loss: {:.7f} / {:.7f}, Auc: {:.4f} / {:.4f}, Acc: {:.4f} / {:.4f}, F1: {:.4f} /'
            ' {:.4f} '.format(outer_split_no, inner_split_no, epoch, train_metrics['loss'], val_metrics['loss'],
                              train_metrics['auc'], val_metrics['auc'],
                              train_metrics['acc'], val_metrics['acc'],
                              train_metrics['f1'], val_metrics['f1']))
        wandb.log({
            f'train_loss{inner_split_no}': train_metrics['loss'], f'val_loss{inner_split_no}': val_metrics['loss'],
            f'train_auc{inner_split_no}': train_metrics['auc'], f'val_auc{inner_split_no}': val_metrics['auc'],
            f'train_acc{inner_split_no}': train_metrics['acc'], f'val_acc{inner_split_no}': val_metrics['acc'],
            f'train_sens{inner_split_no}': train_metrics['sensitivity'],
            f'val_sens{inner_split_no}': val_metrics['sensitivity'],
            f'train_spec{inner_split_no}': train_metrics['specificity'],
            f'val_spec{inner_split_no}': val_metrics['specificity'],
            f'train_f1{inner_split_no}': train_metrics['f1'], f'val_f1{inner_split_no}': val_metrics['f1']
        })
    else:
        print(
            '{:1d}-{:1d}-Epoch: {:03d}, Loss: {:.7f} / {:.7f}, R2: {:.4f} / {:.4f}, R: {:.4f} / {:.4f}'
            ''.format(outer_split_no, inner_split_no, epoch, train_metrics['loss'], val_metrics['loss'],
                      train_metrics['r2'], val_metrics['r2'],
                      train_metrics['r'], val_metrics['r']))
        wandb.log({
            f'train_loss{inner_split_no}': train_metrics['loss'], f'val_loss{inner_split_no}': val_metrics['loss'],
            f'train_r2{inner_split_no}': train_metrics['r2'], f'val_r2{inner_split_no}': val_metrics['r2'],
            f'train_r{inner_split_no}': train_metrics['r'], f'val_r{inner_split_no}': val_metrics['r']
        })

    if run_cfg['param_pooling'] in [PoolingStrategy.DIFFPOOL, PoolingStrategy.DP_MAX, PoolingStrategy.DP_ADD, PoolingStrategy.DP_MEAN]:
        wandb.log({
            f'train_link_loss{inner_split_no}': link_loss, f'val_link_loss{inner_split_no}': val_metrics['link_loss'],
            f'train_ent_loss{inner_split_no}': ent_loss, f'val_ent_loss{inner_split_no}': val_metrics['ent_loss']
        })

    return val_metrics


def create_fold_generator(dataset: BrainDataset, run_cfg: Dict[str, Any], num_splits: int):
    if run_cfg['dataset_type'] == DatasetType.HCP:
        # Stratification will occur with regards to both the sex and session day
        skf = StratifiedGroupKFold(n_splits=num_splits, random_state=1111)
        merged_labels = merge_y_and_others(torch.cat([data.y for data in dataset], dim=0),
                                           torch.cat([data.index for data in dataset], dim=0))
        skf_generator = skf.split(np.zeros((len(dataset), 1)),
                                  merged_labels,
                                  groups=[data.hcp_id.item() for data in dataset])
    else:
        # UKB stratification over sex, age, and BMI (needs discretisation first)
        sexes = []
        bmis = []
        ages = []
        for data in dataset:
            if run_cfg['analysis_type'] == AnalysisType.FLATTEN_CORRS:
                sexes.append(data.sex.item())
                ages.append(data.age.item())
                bmis.append(data.bmi.item())
            elif run_cfg['target_var'] == 'gender':
                sexes.append(data.y.item())
                ages.append(data.age.item())
                bmis.append(data.bmi.item())
            elif run_cfg['target_var'] == 'age':
                sexes.append(data.sex.item())
                ages.append(data.y.item())
                bmis.append(data.bmi.item())
            elif run_cfg['target_var'] == 'bmi':
                sexes.append(data.sex.item())
                ages.append(data.age.item())
                bmis.append(data.y.item())
        bmis = pd.qcut(bmis, 7, labels=False)
        bmis[np.isnan(bmis)] = 7
        ages = pd.qcut(ages, 7, labels=False)
        strat_labels = LabelEncoder().fit_transform([f'{sexes[i]}{ages[i]}{bmis[i]}' for i in range(len(dataset))])

        skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=1111)
        skf_generator = skf.split(np.zeros((len(dataset), 1)),
                                  strat_labels)

    return skf_generator


def generate_dataset(run_cfg: Dict[str, Any]) -> Union[BrainDataset, FlattenCorrsDataset]:
    if run_cfg['analysis_type'] == AnalysisType.FLATTEN_CORRS:
        name_dataset = create_name_for_flattencorrs_dataset(run_cfg)
        print("Going for", name_dataset)
        dataset = FlattenCorrsDataset(root=name_dataset,
                                      num_nodes=run_cfg['num_nodes'],
                                      connectivity_type=run_cfg['param_conn_type'],
                                      analysis_type=run_cfg['analysis_type'],
                                      dataset_type=run_cfg['dataset_type'],
                                      time_length=run_cfg['time_length'])
    else:
        name_dataset = create_name_for_brain_dataset(num_nodes=run_cfg['num_nodes'],
                                                     time_length=run_cfg['time_length'],
                                                     target_var=run_cfg['target_var'],
                                                     threshold=run_cfg['param_threshold'],
                                                     normalisation=run_cfg['param_normalisation'],
                                                     connectivity_type=run_cfg['param_conn_type'],
                                                     analysis_type=run_cfg['analysis_type'],
                                                     encoding_strategy=run_cfg['param_encoding_strategy'],
                                                     dataset_type=run_cfg['dataset_type'],
                                                     edge_weights=run_cfg['edge_weights'])
        print("Going for", name_dataset)
        class_dataset = HCPDataset if run_cfg['dataset_type'] == DatasetType.HCP else UKBDataset
        dataset = class_dataset(root=name_dataset,
                                target_var=run_cfg['target_var'],
                                num_nodes=run_cfg['num_nodes'],
                                threshold=run_cfg['param_threshold'],
                                connectivity_type=run_cfg['param_conn_type'],
                                normalisation=run_cfg['param_normalisation'],
                                analysis_type=run_cfg['analysis_type'],
                                encoding_strategy=run_cfg['param_encoding_strategy'],
                                time_length=run_cfg['time_length'],
                                edge_weights=run_cfg['edge_weights'])

    return dataset


def generate_xgb_model(run_cfg: Dict[str, Any]) -> XGBModel:
    if run_cfg['target_var'] == 'gender':
        model = XGBClassifier(subsample=run_cfg['subsample'],
                              learning_rate=run_cfg['learning_rate'],
                              max_depth=run_cfg['max_depth'],
                              min_child_weight=run_cfg['min_child_weight'],
                              colsample_bytree=run_cfg['colsample_bytree'],
                              colsample_bynode=run_cfg['colsample_bynode'],
                              colsample_bylevel=run_cfg['colsample_bylevel'],
                              n_estimators=run_cfg['n_estimators'],
                              gamma=run_cfg['gamma'],
                              n_jobs=-1,
                              random_state=1111)
    else:
        model = XGBRegressor(subsample=run_cfg['subsample'],
                             learning_rate=run_cfg['learning_rate'],
                             max_depth=run_cfg['max_depth'],
                             min_child_weight=run_cfg['min_child_weight'],
                             colsample_bytree=run_cfg['colsample_bytree'],
                             colsample_bynode=run_cfg['colsample_bynode'],
                             colsample_bylevel=run_cfg['colsample_bylevel'],
                             n_estimators=run_cfg['n_estimators'],
                             gamma=run_cfg['gamma'],
                             n_jobs=-1,
                             random_state=1111)
    return model


def generate_st_model(run_cfg: Dict[str, Any], for_test: bool = False) -> SpatioTemporalModel:
    if run_cfg['param_encoding_strategy'] in [EncodingStrategy.NONE, EncodingStrategy.STATS]:
        encoding_model = None
    else:
        if run_cfg['param_encoding_strategy'] == EncodingStrategy.AE3layers:
            pass  # from encoders import AE  # Necessary to torch.load
        elif run_cfg['param_encoding_strategy'] == EncodingStrategy.VAE3layers:
            pass  # from encoders import VAE  # Necessary to torch.load
        encoding_model = torch.load(create_best_encoder_name(ts_length=run_cfg['time_length'],
                                                             outer_split_num=outer_split_num,
                                                             encoder_name=run_cfg['param_encoding_strategy'].value))

    model = SpatioTemporalModel(run_cfg=run_cfg,
                                encoding_model=encoding_model
                                ).to(run_cfg['device_run'])

    if not for_test:
        #wandb.watch(model, log='all')
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of trainable params:", trainable_params)
    # elif analysis_type == AnalysisType.FLATTEN_CORRS or analysis_type == AnalysisType.FLATTEN_CORRS_THRESHOLD:
    #    model = XGBClassifier(n_jobs=-1, seed=1111, random_state=1111, **params)
    return model


def fit_xgb_model(out_fold_num: int, in_fold_num: int, run_cfg: Dict[str, Any], model: XGBModel,
                  X_train_in: FlattenCorrsDataset, X_val_in: FlattenCorrsDataset) -> Dict:
    model_saving_path = create_name_for_xgbmodel(model=model,
                                                 outer_split_num=out_fold_num,
                                                 inner_split_num=in_fold_num,
                                                 run_cfg=run_cfg
                                                 )

    train_arr = np.array([data.x.numpy() for data in X_train_in])
    val_arr = np.array([data.x.numpy() for data in X_val_in])

    if run_cfg['target_var'] == 'gender':
        y_train = [int(data.sex.item()) for data in X_train_in]
        y_val = [int(data.sex.item()) for data in X_val_in]
    elif run_cfg['target_var'] == 'age':
        # np.array() because of printing calls in the regressor_metrics function
        y_train = np.array([float(data.age.item()) for data in X_train_in])
        y_val = np.array([float(data.age.item()) for data in X_val_in])

    model.fit(train_arr, y_train, callbacks=[wandb.xgboost.wandb_callback()])

    pickle.dump(model, open(model_saving_path, "wb"))

    if run_cfg['target_var'] == 'gender':
        train_metrics = return_classifier_metrics(y_train,
                                                  pred_prob=model.predict_proba(train_arr)[:, 1],
                                                  pred_binary=model.predict(train_arr),
                                                  flatten_approach=True)
        val_metrics = return_classifier_metrics(y_val,
                                                pred_prob=model.predict_proba(val_arr)[:, 1],
                                                pred_binary=model.predict(val_arr),
                                                flatten_approach=True)

        print('{:1d}-{:1d}: Auc: {:.4f} / {:.4f}, Acc: {:.4f} / {:.4f}, F1: {:.4f} /'
              ' {:.4f} '.format(out_fold_num, in_fold_num,
                                train_metrics['auc'], val_metrics['auc'],
                                train_metrics['acc'], val_metrics['acc'],
                                train_metrics['f1'], val_metrics['f1']))
        wandb.log({
            f'train_auc{in_fold_num}': train_metrics['auc'], f'val_auc{in_fold_num}': val_metrics['auc'],
            f'train_acc{in_fold_num}': train_metrics['acc'], f'val_acc{in_fold_num}': val_metrics['acc'],
            f'train_sens{in_fold_num}': train_metrics['sensitivity'],
            f'val_sens{in_fold_num}': val_metrics['sensitivity'],
            f'train_spec{in_fold_num}': train_metrics['specificity'],
            f'val_spec{in_fold_num}': val_metrics['specificity'],
            f'train_f1{in_fold_num}': train_metrics['f1'], f'val_f1{in_fold_num}': val_metrics['f1']
        })
    else:
        train_metrics = return_regressor_metrics(y_train,
                                                 pred_prob=model.predict(train_arr))
        val_metrics = return_regressor_metrics(y_val,
                                               pred_prob=model.predict(val_arr))

        print('{:1d}-{:1d}: R2: {:.4f} / {:.4f}, R: {:.4f} / {:.4f}'.format(out_fold_num, in_fold_num,
                                                                            train_metrics['r2'], val_metrics['r2'],
                                                                            train_metrics['r'], val_metrics['r']))
        wandb.log({
            f'train_r2{in_fold_num}': train_metrics['r2'], f'val_r2{in_fold_num}': val_metrics['r2'],
            f'train_r{in_fold_num}': train_metrics['r'], f'val_r{in_fold_num}': val_metrics['r']
        })

    return val_metrics


def fit_st_model(out_fold_num: int, in_fold_num: int, run_cfg: Dict[str, Any], model: SpatioTemporalModel,
                 X_train_in: BrainDataset, X_val_in: BrainDataset, label_scaler: MinMaxScaler = None) -> Dict:
    train_in_loader = DataLoader(X_train_in, batch_size=run_cfg['batch_size'], shuffle=True)#, **kwargs_dataloader)
    val_loader = DataLoader(X_val_in, batch_size=run_cfg['batch_size'], shuffle=False)#, **kwargs_dataloader)

    ###########
    ## OPTIMISER
    ###########
    if run_cfg['optimiser'] == Optimiser.SGD:
        print('--> OPTIMISER: SGD')
        optimizer = torch.optim.SGD(model.parameters(),
                                     lr=run_cfg['param_lr'],
                                     weight_decay=run_cfg['param_weight_decay'],
                                    )
    elif run_cfg['optimiser'] == Optimiser.ADAM:
        print('--> OPTIMISER: Adam')
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=run_cfg['param_lr'],
                                     weight_decay=run_cfg['param_weight_decay'])
    elif run_cfg['optimiser'] == Optimiser.ADAMW:
        print('--> OPTIMISER: AdamW')
        optimizer = torch.optim.AdamW(model.parameters(),
                                     lr=run_cfg['param_lr'],
                                     weight_decay=run_cfg['param_weight_decay'])
    elif run_cfg['optimiser'] == Optimiser.RMSPROP:
        print('--> OPTIMISER: RMSprop')
        optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=run_cfg['param_lr'],
                                        weight_decay=run_cfg['param_weight_decay'])

    ###########
    ## LR SCHEDULER
    ###########
    if run_cfg['lr_scheduler'] == LRScheduler.STEP:
        print('--> LR SCHEDULER: Step')
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(run_cfg['num_epochs'] / 5), gamma=0.1, verbose=True)
    elif run_cfg['lr_scheduler'] == LRScheduler.PLATEAU:
        print('--> LR SCHEDULER: Plateau')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               patience=run_cfg['early_stop_steps']-2,
                                                               verbose=True)
    elif run_cfg['lr_scheduler'] == LRScheduler.COS_ANNEALING:
        print('--> LR SCHEDULER: Cosine Annealing')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=run_cfg['num_epochs'], verbose=True)
    elif run_cfg['lr_scheduler'] == LRScheduler.NONE:
        print('--> LR SCHEDULER: None')

    ###########
    ## EARLY STOPPING
    ###########
    model_saving_name = create_name_for_model(run_cfg=run_cfg,
                                              model=model,
                                              outer_split_num=out_fold_num,
                                              inner_split_num=in_fold_num,
                                              prefix_location=''
                                              )

    early_stopping = EarlyStopping(patience=run_cfg['early_stop_steps'], model_saving_name=model_saving_name)

    ###########
    ## Model Exponential Moving Average (EMA)
    ###########
    if run_cfg['use_ema']:
        print('--> USING EMA (some repetitions on models outputs because of copying)!')
        # Issues with deepcopy() when using weightnorm, therefore, this workaround is needed
        new_model = generate_st_model(run_cfg, for_test=True)
        new_model.load_state_dict(model.state_dict())
        model_ema = ModelEmaV2(new_model)
    else:
        model_ema = None
    # Only after knowing EMA deep copies "model", I call wandb.watch()
    wandb.watch(model, log='all')

    for epoch in range(run_cfg['num_epochs'] + 1):
        val_metrics = training_step(out_fold_num,
                                    in_fold_num,
                                    epoch,
                                    model,
                                    train_in_loader,
                                    val_loader,
                                    optimizer,
                                    model_ema=model_ema,
                                    run_cfg=run_cfg,
                                    label_scaler=label_scaler)

        # Calling early_stopping() to update best metrics and stoppping state
        if model_ema is not None:
            early_stopping(val_metrics, model_ema.module, label_scaler)
        else:
            early_stopping(val_metrics, model, label_scaler)
        if early_stopping.early_stop:
            print("EARLY STOPPING IT")
            break

        if run_cfg['lr_scheduler'] in [LRScheduler.STEP, LRScheduler.COS_ANNEALING]:
            scheduler.step()
        elif run_cfg['lr_scheduler'] == LRScheduler.PLATEAU:
            scheduler.step(val_metrics['loss'])

    # wandb.unwatch()
    return early_stopping.best_model_metrics


def get_empty_metrics_dict(run_cfg: Dict[str, Any]) -> Dict[str, list]:
    if run_cfg['target_var'] == 'gender':
        tmp_dict = {'loss': [], 'sensitivity': [], 'specificity': [], 'acc': [], 'f1': [], 'auc': [],
                    'ent_loss': [], 'link_loss': [], 'best_epoch': []}
    else:
        tmp_dict = {'loss': [], 'r2': [], 'r': [], 'ent_loss': [], 'link_loss': []}
    return tmp_dict


def send_inner_loop_metrics_to_wandb(overall_metrics: Dict[str, list]):
    for key, values in overall_metrics.items():
        if len(values) == 0 or values[0] is None:
            continue
        elif len(values) == 1:
            wandb.run.summary[f"mean_val_{key}"] = values[0]
        else:
            wandb.run.summary[f"mean_val_{key}"] = np.mean(values)
            wandb.run.summary[f"std_val_{key}"] = np.std(values)
            wandb.run.summary[f"values_val_{key}"] = values


def update_overall_metrics(overall_metrics: Dict[str, list], inner_fold_metrics: Dict[str, float]):
    for key, value in inner_fold_metrics.items():
        overall_metrics[key].append(value)


def send_global_results(test_metrics: Dict[str, float]):
    for key, value in test_metrics.items():
        wandb.run.summary[f"values_test_{key}"] = value


if __name__ == '__main__':
    # Because of strange bug with symbolic links in server
    os.environ['WANDB_DISABLE_CODE'] = 'true'

    wandb.init(project='st_extra', save_code=True)#, dir='/work1/tiago/wandb')
    config = wandb.config

    print('Config file from wandb:', config)

    torch.manual_seed(1)
    np.random.seed(1111)
    random.seed(1111)
    torch.cuda.manual_seed_all(1111)

    # Making a single variable for each argument
    run_cfg: Dict[str, Any] = {
        'analysis_type': AnalysisType(config.analysis_type),
        'dataset_type': DatasetType(config.dataset_type),
        'num_nodes': config.num_nodes,
        'param_conn_type': ConnType(config.conn_type),
        'split_to_test': config.fold_num,
        'target_var': config.target_var,
        'time_length': config.time_length,
    }
    if run_cfg['analysis_type'] in [AnalysisType.ST_UNIMODAL, AnalysisType.ST_MULTIMODAL, AnalysisType.ST_UNIMODAL_AVG, AnalysisType.ST_MULTIMODAL_AVG]:
        run_cfg['batch_size'] = config.batch_size
        run_cfg['device_run'] = f'cuda:{get_freer_gpu()}'
        run_cfg['early_stop_steps'] = config.early_stop_steps
        run_cfg['edge_weights'] = config.edge_weights
        run_cfg['model_with_sigmoid'] = True
        run_cfg['num_epochs'] = config.num_epochs
        run_cfg['param_activation'] = config.activation
        run_cfg['param_channels_conv'] = config.channels_conv
        run_cfg['param_conv_strategy'] = ConvStrategy(config.conv_strategy)
        run_cfg['param_dropout'] = config.dropout
        run_cfg['param_encoding_strategy'] = EncodingStrategy(config.encoding_strategy)
        run_cfg['param_lr'] = config.lr
        run_cfg['param_normalisation'] = Normalisation(config.normalisation)
        run_cfg['param_num_gnn_layers'] = config.num_gnn_layers
        run_cfg['param_pooling'] = PoolingStrategy(config.pooling)
        run_cfg['param_threshold'] = config.threshold
        run_cfg['param_weight_decay'] = config.weight_decay
        run_cfg['sweep_type'] = SweepType(config.sweep_type)
        run_cfg['temporal_embed_size'] = config.temporal_embed_size

        run_cfg['ts_spit_num'] = int(4800 / run_cfg['time_length'])

        # Not sure whether this makes a difference with the cuda random issues, but it was in the examples :(
        #kwargs_dataloader = {'num_workers': 1, 'pin_memory': True} if run_cfg['device_run'].startswith('cuda') else {}

        # Definitions depending on sweep_type
        run_cfg['param_gat_heads'] = 0
        if run_cfg['sweep_type'] == SweepType.GAT:
            run_cfg['param_gat_heads'] = config.gat_heads


        # TCN components
        run_cfg['tcn_depth'] = config.tcn_depth
        run_cfg['tcn_kernel'] = config.tcn_kernel
        run_cfg['tcn_hidden_units'] = config.tcn_hidden_units
        run_cfg['tcn_final_transform_layers'] = config.tcn_final_transform_layers
        run_cfg['tcn_norm_strategy'] = config.tcn_norm_strategy

        # Training characteristics
        run_cfg['lr_scheduler'] = LRScheduler(config.lr_scheduler)
        run_cfg['optimiser'] = Optimiser(config.optimiser)
        run_cfg['use_ema'] = config.use_ema

        # Node model and final hyperparameters
        run_cfg['nodemodel_aggr'] = config.nodemodel_aggr
        run_cfg['nodemodel_scalers'] = config.nodemodel_scalers
        run_cfg['nodemodel_layers'] = config.nodemodel_layers
        run_cfg['final_mlp_layers'] = config.final_mlp_layers

    elif run_cfg['analysis_type'] in [AnalysisType.FLATTEN_CORRS]:
        run_cfg['device_run'] = 'cpu'
        run_cfg['colsample_bylevel'] = config.colsample_bylevel
        run_cfg['colsample_bynode'] = config.colsample_bynode
        run_cfg['colsample_bytree'] = config.colsample_bytree
        run_cfg['gamma'] = config.gamma
        run_cfg['learning_rate'] = config.learning_rate
        run_cfg['max_depth'] = config.max_depth
        run_cfg['min_child_weight'] = config.min_child_weight
        run_cfg['n_estimators'] = config.n_estimators
        run_cfg['subsample'] = config.subsample

    N_OUT_SPLITS: int = 5
    N_INNER_SPLITS: int = 5

    # Handling inputs and what is possible
    if run_cfg['analysis_type'] not in [AnalysisType.ST_MULTIMODAL, AnalysisType.ST_UNIMODAL,
                                        AnalysisType.ST_MULTIMODAL_AVG, AnalysisType.ST_UNIMODAL_AVG,
                                        AnalysisType.FLATTEN_CORRS]:
        print('Not yet ready for this analysis type at the moment')
        exit(-1)

    print('This run will not be deterministic')
    if run_cfg['target_var'] not in ['gender', 'age', 'bmi']:
        print('Unrecognised target_var')
        exit(-1)

    run_cfg['multimodal_size'] = 0
    #if run_cfg['analysis_type'] == AnalysisType.ST_MULTIMODAL:
    #    run_cfg['multimodal_size'] = 10
    #elif run_cfg['analysis_type'] == AnalysisType.ST_UNIMODAL:
    #    run_cfg['multimodal_size'] = 0

    if run_cfg['target_var'] in ['age', 'bmi']:
        run_cfg['model_with_sigmoid'] = False

    print('Resulting run_cfg:', run_cfg)
    # DATASET
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

    scaler_labels = None
    # Scaling for regression problem
    if run_cfg['analysis_type'] in [AnalysisType.ST_UNIMODAL, AnalysisType.ST_MULTIMODAL] and \
            run_cfg['target_var'] in ['age', 'bmi']:
        print('Mean of distribution BEFORE scaling:', np.mean([data.y.item() for data in X_train_out]),
              '/', np.mean([data.y.item() for data in X_test_out]))
        scaler_labels = MinMaxScaler().fit(np.array([data.y.item() for data in X_train_out]).reshape(-1, 1))
        for elem in X_train_out:
            elem.y[0] = scaler_labels.transform([elem.y.numpy()])[0, 0]
        for elem in X_test_out:
            elem.y[0] = scaler_labels.transform([elem.y.numpy()])[0, 0]

    # Train / test sets defined, running the rest
    print('Size is:', len(X_train_out), '/', len(X_test_out))
    if run_cfg['analysis_type'] == AnalysisType.FLATTEN_CORRS:
        print('Positive sex classes:', sum([data.sex.item() for data in X_train_out]),
              '/', sum([data.sex.item() for data in X_test_out]))
        print('Mean age distribution:', np.mean([data.age.item() for data in X_train_out]),
              '/', np.mean([data.age.item() for data in X_test_out]))
    elif run_cfg['target_var'] in ['age', 'bmi']:
        print('Mean of distribution', np.mean([data.y.item() for data in X_train_out]),
              '/', np.mean([data.y.item() for data in X_test_out]))
    else:  # target_var == gender
        print('Positive classes:', sum([data.y.item() for data in X_train_out]),
              '/', sum([data.y.item() for data in X_test_out]))

    skf_inner_generator = create_fold_generator(X_train_out, run_cfg, N_INNER_SPLITS)

    #################
    # Main inner-loop
    #################
    overall_metrics: Dict[str, list] = get_empty_metrics_dict(run_cfg)
    inner_loop_run: int = 0
    for inner_train_index, inner_val_index in skf_inner_generator:
        inner_loop_run += 1

        X_train_in = X_train_out[torch.tensor(inner_train_index)]
        X_val_in = X_train_out[torch.tensor(inner_val_index)]
        print("Inner Size is:", len(X_train_in), "/", len(X_val_in))
        if run_cfg['analysis_type'] == AnalysisType.FLATTEN_CORRS:
            print("Inner Positive sex classes:", sum([data.sex.item() for data in X_train_in]),
                  "/", sum([data.sex.item() for data in X_val_in]))
            print('Mean age distribution:', np.mean([data.age.item() for data in X_train_in]),
                  '/', np.mean([data.age.item() for data in X_val_in]))
        elif run_cfg['target_var'] in ['age', 'bmi']:
            print('Mean of distribution', np.mean([data.y.item() for data in X_train_in]),
                  '/', np.mean([data.y.item() for data in X_val_in]))
        else:
            print("Inner Positive classes:", sum([data.y.item() for data in X_train_in]),
                  "/", sum([data.y.item() for data in X_val_in]))

        run_cfg['dataset_indegree'] = calculate_indegree_histogram(X_train_in)
        print(f'--> Indegree distribution: {run_cfg["dataset_indegree"]}')

        if run_cfg['analysis_type'] in [AnalysisType.ST_UNIMODAL, AnalysisType.ST_MULTIMODAL, AnalysisType.ST_UNIMODAL_AVG, AnalysisType.ST_MULTIMODAL_AVG]:
            model: SpatioTemporalModel = generate_st_model(run_cfg)
        elif run_cfg['analysis_type'] in [AnalysisType.FLATTEN_CORRS]:
            model: XGBModel = generate_xgb_model(run_cfg)
        else:
            model = None

        if run_cfg['analysis_type'] in [AnalysisType.ST_UNIMODAL, AnalysisType.ST_MULTIMODAL, AnalysisType.ST_UNIMODAL_AVG, AnalysisType.ST_MULTIMODAL_AVG]:
            inner_fold_metrics = fit_st_model(out_fold_num=run_cfg['split_to_test'],
                                              in_fold_num=inner_loop_run,
                                              run_cfg=run_cfg,
                                              model=model,
                                              X_train_in=X_train_in,
                                              X_val_in=X_val_in,
                                              label_scaler=scaler_labels)

        elif run_cfg['analysis_type'] in [AnalysisType.FLATTEN_CORRS]:
            inner_fold_metrics = fit_xgb_model(out_fold_num=run_cfg['split_to_test'],
                                               in_fold_num=inner_loop_run,
                                               run_cfg=run_cfg,
                                               model=model,
                                               X_train_in=X_train_in,
                                               X_val_in=X_val_in)
        update_overall_metrics(overall_metrics, inner_fold_metrics)

        # One inner loop only
        #if run_cfg['dataset_type'] == DatasetType.UKB and run_cfg['analysis_type'] in [AnalysisType.ST_UNIMODAL,
        #                                                                               AnalysisType.ST_MULTIMODAL]:
        #    break
        # One inner loop no matter what analysis type for more systematic comparison
        break

    send_inner_loop_metrics_to_wandb(overall_metrics)
    print('Overall inner loop results:', overall_metrics)

    #############################################
    # Final metrics on test set, calculated already for being easy to get the metrics on the best model later
    # Getting best model of the run
    inner_fold_for_val: int = 1
    if run_cfg['analysis_type'] in [AnalysisType.ST_UNIMODAL, AnalysisType.ST_MULTIMODAL, AnalysisType.ST_UNIMODAL_AVG, AnalysisType.ST_MULTIMODAL_AVG]:
        model: SpatioTemporalModel = generate_st_model(run_cfg, for_test=True)

        model_saving_path: str = create_name_for_model(run_cfg=run_cfg,
                                                       model=model,
                                                       outer_split_num=run_cfg['split_to_test'],
                                                       inner_split_num=inner_fold_for_val,
                                                       prefix_location='')

        model.load_state_dict(torch.load(os.path.join(wandb.run.dir, model_saving_path)))
        model.eval()

        # Calculating on test set
        test_out_loader = DataLoader(X_test_out, batch_size=run_cfg['batch_size'], shuffle=False)#, **kwargs_dataloader)

        test_metrics = evaluate_model(model, test_out_loader, run_cfg=run_cfg, label_scaler=scaler_labels)
        print(test_metrics)

        if scaler_labels is None:
            print('{:1d}-Final: {:.7f}, Auc: {:.4f}, Acc: {:.4f}, Sens: {:.4f}, Speci: {:.4f}'
                  ''.format(outer_split_num, test_metrics['loss'], test_metrics['auc'], test_metrics['acc'],
                            test_metrics['sensitivity'], test_metrics['specificity']))
        else:
            print('{:1d}-Final: {:.7f}, R2: {:.4f}, R: {:.4f}'
                  ''.format(outer_split_num, test_metrics['loss'], test_metrics['r2'], test_metrics['r']))
    elif run_cfg['analysis_type'] in [AnalysisType.FLATTEN_CORRS]:
        model: XGBModel = generate_xgb_model(run_cfg)
        model_saving_path = create_name_for_xgbmodel(model=model,
                                                     outer_split_num=run_cfg['split_to_test'],
                                                     inner_split_num=inner_fold_for_val,
                                                     run_cfg=run_cfg
                                                     )
        model = pickle.load(open(model_saving_path, "rb"))
        test_arr = np.array([data.x.numpy() for data in X_test_out])

        if run_cfg['target_var'] == 'gender':
            y_test = [int(data.sex.item()) for data in X_test_out]
            test_metrics = return_classifier_metrics(y_test,
                                                     pred_prob=model.predict_proba(test_arr)[:, 1],
                                                     pred_binary=model.predict(test_arr),
                                                     flatten_approach=True)
            print(test_metrics)

            print('{:1d}-Final: Auc: {:.4f}, Acc: {:.4f}, Sens: {:.4f}, Speci: {:.4f}'
                  ''.format(outer_split_num, test_metrics['auc'], test_metrics['acc'],
                            test_metrics['sensitivity'], test_metrics['specificity']))
        elif run_cfg['target_var'] == 'age':
            # np.array() because of printing calls in the regressor_metrics function
            y_test = np.array([float(data.age.item()) for data in X_test_out])
            test_metrics = return_regressor_metrics(y_test,
                                                    pred_prob=model.predict(test_arr))
            print(test_metrics)
            print('{:1d}-Final: R2: {:.4f}, R: {:.4f}'.format(outer_split_num,
                                                              test_metrics['r2'],
                                                              test_metrics['r']))

    send_global_results(test_metrics)

    #if run_cfg['device_run'] == 'cuda:0':
    #    free_gpu_info()
