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
    SweepType, DatasetType, get_freer_gpu


def train_classifier(model, train_loader, optimizer, pooling_mechanism, device):
    model.train()
    loss_all = 0
    loss_all_link = 0
    loss_all_ent = 0
    criterion = torch.nn.BCELoss()

    grads = {'final_l': [],
             'conv1d_1': []
             }
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        if pooling_mechanism == PoolingStrategy.DIFFPOOL:
            output_batch, link_loss, ent_loss = model(data)
            loss = criterion(output_batch, data.y.unsqueeze(1)) + link_loss + ent_loss
            loss_b_link = link_loss
            loss_b_ent = ent_loss
        else:
            output_batch = model(data)
            loss = criterion(output_batch, data.y.unsqueeze(1))

        loss.backward()

        grads['final_l'].extend(model.final_linear.weight.grad.flatten().cpu().tolist())
        grads['conv1d_1'].extend(model.final_linear.weight.grad.flatten().cpu().tolist())

        loss_all += loss.item() * data.num_graphs
        if pooling_mechanism == PoolingStrategy.DIFFPOOL:
            loss_all_link += loss_b_link.item() * data.num_graphs
            loss_all_ent += loss_b_ent.item() * data.num_graphs
        optimizer.step()
    print("GRAD", np.mean(grads['final_l']), np.max(grads['final_l']), np.std(grads['final_l']))
    # len(train_loader) gives the number of batches
    # len(train_loader.dataset) gives the number of graphs

    # Returning a weighted average according to number of graphs
    return loss_all / len(train_loader.dataset), loss_all_link / len(train_loader.dataset), loss_all_ent / len(
        train_loader.dataset)


def return_metrics(labels, pred_binary, pred_prob, loss_value=None, link_loss_value=None, ent_loss_value=None):
    roc_auc = roc_auc_score(labels, pred_prob)
    acc = accuracy_score(labels, pred_binary)
    f1 = f1_score(labels, pred_binary, zero_division=0)
    report = classification_report(labels, pred_binary, output_dict=True, zero_division=0)
    sens = report['1.0']['recall']
    spec = report['0.0']['recall']

    return {'loss': loss_value,
            'link_loss': link_loss_value,
            'ent_loss': ent_loss_value,
            'auc': roc_auc,
            'acc': acc,
            'f1': f1,
            'sensitivity': sens,
            'specificity': spec
            }


def evaluate_classifier(model, loader, pooling_mechanism, device):
    model.eval()
    criterion = torch.nn.BCELoss()

    predictions = []
    labels = []
    test_error = 0
    test_link_loss = 0
    test_ent_loss = 0

    for data in loader:
        with torch.no_grad():
            data = data.to(device)
            if pooling_mechanism == PoolingStrategy.DIFFPOOL:
                output_batch, link_loss, ent_loss = model(data)
                output_batch = output_batch.flatten()
                loss = criterion(output_batch, data.y) + link_loss + ent_loss
                loss_b_link = link_loss
                loss_b_ent = ent_loss
            else:
                output_batch = model(data)
                output_batch = output_batch.flatten()
                loss = criterion(output_batch, data.y)

            test_error += loss.item() * data.num_graphs
            if pooling_mechanism == PoolingStrategy.DIFFPOOL:
                test_link_loss += loss_b_link.item() * data.num_graphs
                test_ent_loss += loss_b_ent.item() * data.num_graphs

            pred = output_batch.detach().cpu().numpy()

            label = data.y.detach().cpu().numpy()
            predictions.append(pred)
            labels.append(label)
    predictions = np.hstack(predictions)
    labels = np.hstack(labels)

    # if save_path_preds is not None:
    #    np.save('results/l_' + save_path_preds, labels)
    #    np.save('results/p_' + save_path_preds, predictions)

    pred_binary = np.where(predictions > 0.5, 1, 0)

    return return_metrics(labels, pred_binary, predictions,
                          loss_value=test_error / len(loader.dataset),
                          link_loss_value=test_link_loss / len(loader.dataset),
                          ent_loss_value=test_ent_loss / len(loader.dataset))


def classifier_step(outer_split_no, inner_split_no, epoch, model, train_loader, val_loader, optimizer,
                    pooling_mechanism, device):
    loss, link_loss, ent_loss = train_classifier(model, train_loader, optimizer, pooling_mechanism, device)
    train_metrics = evaluate_classifier(model, train_loader, pooling_mechanism, device)
    val_metrics = evaluate_classifier(model, val_loader, pooling_mechanism, device)

    print('{:1d}-{:1d}-Epoch: {:03d}, Loss: {:.7f} / {:.7f}, Auc: {:.4f} / {:.4f}, Acc: {:.4f} / {:.4f}, F1: {:.4f} /'
          ' {:.4f} '.format(outer_split_no, inner_split_no, epoch, loss, val_metrics['loss'],
                            train_metrics['auc'], val_metrics['auc'],
                            train_metrics['acc'], val_metrics['acc'],
                            train_metrics['f1'], val_metrics['f1']))
    wandb.log({
        f'train_loss{inner_split_no}': loss, f'val_loss{inner_split_no}': val_metrics['loss'],
        f'train_auc{inner_split_no}': train_metrics['auc'], f'val_auc{inner_split_no}': val_metrics['auc'],
        f'train_acc{inner_split_no}': train_metrics['acc'], f'val_acc{inner_split_no}': val_metrics['acc'],
        f'train_sens{inner_split_no}': train_metrics['sensitivity'],
        f'val_sens{inner_split_no}': val_metrics['sensitivity'],
        f'train_spec{inner_split_no}': train_metrics['specificity'],
        f'val_spec{inner_split_no}': val_metrics['specificity'],
        f'train_f1{inner_split_no}': train_metrics['f1'], f'val_f1{inner_split_no}': val_metrics['f1']
    })
    if pooling_mechanism == PoolingStrategy.DIFFPOOL:
        wandb.log({
            'train_link_loss': link_loss, 'val_link_loss': val_metrics['link_loss'],
            'train_ent_loss': ent_loss, 'val_ent_loss': val_metrics['ent_loss']
        })

    return val_metrics


def get_array_data(flatten_correlations, data_fold, num_nodes=50):
    tmp_array = []
    tmp_y = []

    for d in data_fold:
        if num_nodes == 376:
            tmp_array.append(flatten_correlations[d.ukb_id.item()])
        else:
            tmp_array.append(flatten_correlations[(d.hcp_id.item(), d.index.item())])
        tmp_y.append(d.y.item())

    return np.array(tmp_array), np.array(tmp_y)


def create_fold_generator(dataset: BrainDataset, dataset_type: DatasetType, num_splits: int):
    if dataset_type == DatasetType.HCP:
        # Stratification will occur with regards to both the sex and session day
        skf = StratifiedGroupKFold(n_splits=num_splits, random_state=1111)
        merged_labels = merge_y_and_others(torch.cat([data.y for data in dataset], dim=0),
                                           torch.cat([data.index for data in dataset], dim=0))
        skf_generator = skf.split(np.zeros((len(dataset), 1)),
                                  merged_labels,
                                  groups=[data.hcp_id.item() for data in dataset])
    else:
        skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=1111)
        skf_generator = skf.split(np.zeros((len(dataset), 1)),
                                  np.array([data.y.item() for data in dataset]))

    return skf_generator


def generate_dataset(run_cfg: Dict[str, Any]) -> BrainDataset:
    name_dataset = create_name_for_brain_dataset(num_nodes=run_cfg['num_nodes'],
                                                 time_length=run_cfg['time_length'],
                                                 target_var=run_cfg['target_var'],
                                                 threshold=run_cfg['param_threshold'],
                                                 normalisation=run_cfg['param_normalisation'],
                                                 connectivity_type=run_cfg['param_conn_type'],
                                                 analysis_type=run_cfg['analysis_type'],
                                                 dataset_type=run_cfg['dataset_type'])
    print("Going for", name_dataset)
    class_dataset = HCPDataset if run_cfg['dataset_type'] == DatasetType.HCP else UKBDataset
    dataset = class_dataset(root=name_dataset,
                            target_var=run_cfg['target_var'],
                            num_nodes=run_cfg['num_nodes'],
                            threshold=run_cfg['param_threshold'],
                            connectivity_type=run_cfg['param_conn_type'],
                            normalisation=run_cfg['param_normalisation'],
                            analysis_type=run_cfg['analysis_type'],
                            time_length=run_cfg['time_length'])
    # if run_cfg['analysis_type'] == AnalysisType.FLATTEN_CORRS:
    #    if num_nodes == 376:
    #        flatten_correlations = create_ukb_corrs_flatten()
    #    else:
    #        flatten_correlations = create_hcp_correlation_vals(num_nodes, ts_split_num=ts_spit_num)
    # elif run_cfg['analysis_type'] == AnalysisType.FLATTEN_CORRS_THRESHOLD:
    #    flatten_correlations = create_hcp_correlation_vals(num_nodes, ts_split_num=ts_spit_num,
    #                                                       binarise=True, threshold=param_threshold)
    return dataset


def generate_st_model(run_cfg: Dict[str, Any]) -> SpatioTemporalModel:
    if run_cfg['param_encoding_strategy'] != EncodingStrategy.NONE:
        if run_cfg['param_encoding_strategy'] == EncodingStrategy.AE3layers:
            pass  # from encoders import AE  # Necessary to torch.load
        elif run_cfg['param_encoding_strategy'] == EncodingStrategy.VAE3layers:
            pass  # from encoders import VAE  # Necessary to torch.load
        encoding_model = torch.load(create_best_encoder_name(ts_length=run_cfg['time_length'],
                                                             outer_split_num=outer_split_num,
                                                             encoder_name=run_cfg['param_encoding_strategy'].value))
    else:
        encoding_model = None
    model = SpatioTemporalModel(num_time_length=run_cfg['time_length'],
                                dropout_perc=run_cfg['param_dropout'],
                                pooling=run_cfg['param_pooling'],
                                channels_conv=run_cfg['param_channels_conv'],
                                activation=run_cfg['param_activation'],
                                conv_strategy=run_cfg['param_conv_strategy'],
                                add_gat=run_cfg['param_add_gat'],
                                gat_heads=run_cfg['param_gat_heads'],
                                add_gcn=run_cfg['param_add_gcn'],
                                final_sigmoid=run_cfg['model_with_sigmoid'],
                                num_nodes=run_cfg['num_nodes'],
                                num_gnn_layers=run_cfg['param_num_gnn_layers'],
                                encoding_model=encoding_model,
                                multimodal_size=run_cfg['multimodal_size']
                                ).to(run_cfg['device_run'])
    wandb.watch(model, log='all')
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable params:", trainable_params)
    # elif analysis_type == AnalysisType.FLATTEN_CORRS or analysis_type == AnalysisType.FLATTEN_CORRS_THRESHOLD:
    #    model = XGBClassifier(n_jobs=-1, seed=1111, random_state=1111, **params)
    return model


def fit_st_model(out_fold_num: int, in_fold_num: int, run_cfg: Dict[str, Any], model: SpatioTemporalModel,
                 X_train_in: BrainDataset, X_val_in: BrainDataset) -> Dict:
    train_in_loader = DataLoader(X_train_in, batch_size=run_cfg['batch_size'], shuffle=True, **kwargs_dataloader)
    val_loader = DataLoader(X_val_in, batch_size=run_cfg['batch_size'], shuffle=False, **kwargs_dataloader)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=run_cfg['param_lr'],
                                 weight_decay=run_cfg['param_weight_decay'])

    model_saving_path = create_name_for_model(target_var=run_cfg['target_var'],
                                              model=model,
                                              outer_split_num=out_fold_num,
                                              inner_split_num=in_fold_num,
                                              n_epochs=run_cfg['num_epochs'],
                                              threshold=run_cfg['param_threshold'],
                                              batch_size=run_cfg['batch_size'],
                                              num_nodes=run_cfg['num_nodes'],
                                              conn_type=run_cfg['param_conn_type'],
                                              normalisation=run_cfg['param_normalisation'],
                                              analysis_type=run_cfg['analysis_type'],
                                              metric_evaluated='loss',
                                              dataset_type=run_cfg['dataset_type'],
                                              lr=run_cfg['param_lr'],
                                              weight_decay=run_cfg['param_weight_decay'])

    best_model_metrics = {'loss': 9999}

    last_losses_val = deque([9999 for _ in range(run_cfg['early_stop_steps'])], maxlen=run_cfg['early_stop_steps'])
    for epoch in range(run_cfg['num_epochs'] + 1):
        val_metrics = classifier_step(out_fold_num,
                                      in_fold_num,
                                      epoch,
                                      model,
                                      train_in_loader,
                                      val_loader,
                                      optimizer,
                                      run_cfg['param_pooling'],
                                      run_cfg['device_run'])
        if sum([val_metrics['loss'] > loss for loss in last_losses_val]) == run_cfg['early_stop_steps']:
            print("EARLY STOPPING IT")
            break
        last_losses_val.append(val_metrics['loss'])

        if val_metrics['loss'] < best_model_metrics['loss']:
            best_model_metrics['loss'] = val_metrics['loss']
            best_model_metrics['sensitivity'] = val_metrics['sensitivity']
            best_model_metrics['specificity'] = val_metrics['specificity']
            best_model_metrics['acc'] = val_metrics['acc']
            best_model_metrics['f1'] = val_metrics['f1']
            best_model_metrics['auc'] = val_metrics['auc']
            if run_cfg['param_pooling'] == PoolingStrategy.DIFFPOOL:
                best_model_metrics['ent_loss'] = val_metrics['ent_loss']
                best_model_metrics['link_loss'] = val_metrics['link_loss']

            # wandb.unwatch()#[model])
            # torch.save(model, model_names['loss'])
            torch.save(model.state_dict(), model_saving_path)
    #wandb.unwatch()
    return best_model_metrics


def get_empty_metrics_dict(pooling_mechanism: PoolingStrategy) -> Dict[str, list]:
    tmp_dict =  {'loss': [], 'sensitivity': [], 'specificity': [], 'acc': [], 'f1': [], 'auc': []}
    if pooling_mechanism == PoolingStrategy.DIFFPOOL:
        tmp_dict['ent_loss'] = []
        tmp_dict['link_loss'] = []

    return tmp_dict


def send_inner_loop_metrics_to_wandb(overall_metrics: Dict[str, list]):
    for key, values in overall_metrics.items():
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

    wandb.init(entity='st-team')
    config = wandb.config
    print('Config file from wandb:', config)

    torch.manual_seed(1)
    np.random.seed(1111)
    random.seed(1111)
    torch.cuda.manual_seed_all(1111)

    # Making a single variable for each argument
    run_cfg: Dict[str, Any] = {
        'device_run': f'cuda:{get_freer_gpu()}',
        'num_epochs': config.num_epochs,
        'target_var': config.target_var,
        'model_with_sigmoid': True,
        'param_activation': config.activation,
        'split_to_test': config.fold_num,
        'batch_size': config.batch_size,
        'num_nodes': config.num_nodes,
        'param_conn_type': ConnType(config.conn_type),
        'param_conv_strategy': ConvStrategy(config.conv_strategy),
        'param_channels_conv': config.channels_conv,
        'param_normalisation': Normalisation(config.normalisation),
        'analysis_type': AnalysisType(config.analysis_type),
        'dataset_type': DatasetType(config.dataset_type),
        'time_length': config.time_length,
        'param_encoding_strategy': EncodingStrategy(config.encoding_strategy),
        'early_stop_steps': config.early_stop_steps,
        'param_dropout': config.dropout,
        'param_weight_decay': config.weight_decay,
        'param_lr': config.lr,
        'param_threshold': config.threshold,
        'param_num_gnn_layers': config.num_gnn_layers,
        'multimodal_size': 10
    }
    run_cfg['ts_spit_num'] = int(4800 / run_cfg['time_length'])

    # Not sure whether this makes a difference with the cuda random issues, but it was in the examples :(
    kwargs_dataloader = {'num_workers': 1, 'pin_memory': True} if run_cfg['device_run'].startswith('cuda') else {}

    # Definitions depending on sweep_type
    run_cfg['param_pooling'] = PoolingStrategy(config.pooling)
    sweep_type = SweepType(config.sweep_type)
    run_cfg['param_gat_heads'] = 0
    run_cfg['param_add_gcn'] = False
    run_cfg['param_add_gat'] = False
    if sweep_type == SweepType.GCN:
        run_cfg['param_add_gcn'] = True
    elif sweep_type == SweepType.GAT:
        run_cfg['param_add_gat'] = True
        run_cfg['param_gat_heads'] = config.gat_heads

    if run_cfg['param_pooling'] == PoolingStrategy.CONCAT:
        run_cfg['batch_size'] -= 50

    N_OUT_SPLITS = 5
    N_INNER_SPLITS = 5

    # Handling inputs and what is possible
    if run_cfg['analysis_type'] not in [AnalysisType.ST_MULTIMODAL]:
        print('Not yet ready for this analysis type at the moment')
        exit(-1)

    print('This run will not be deterministic')
    if run_cfg['target_var'] not in ['gender']:
        print('Unrecognised target_var')
        exit(-1)
    else:
        print('Predicting', run_cfg)

    # DATASET
    dataset = generate_dataset(run_cfg)

    skf_outer_generator = create_fold_generator(dataset, run_cfg['dataset_type'], N_OUT_SPLITS)

    # Getting train / test folds
    outer_split_num = 0
    for train_index, test_index in skf_outer_generator:
        outer_split_num += 1
        # Only run for the specific fold defined in the script arguments.
        if outer_split_num != run_cfg['split_to_test']:
            continue

        X_train_out = dataset[torch.tensor(train_index)]
        X_test_out = dataset[torch.tensor(test_index)]

        break

    # Train / test sets defined, running the rest
    print("Size is:", len(X_train_out), "/", len(X_test_out))
    print("Positive classes:", sum([data.y.item() for data in X_train_out]),
          "/", sum([data.y.item() for data in X_test_out]))

    skf_inner_generator = create_fold_generator(X_train_out, run_cfg['dataset_type'], N_INNER_SPLITS)

    #################
    # Main inner-loop
    #################
    overall_metrics = get_empty_metrics_dict(run_cfg['param_pooling'])
    inner_loop_run = 0
    for inner_train_index, inner_val_index in skf_inner_generator:
        inner_loop_run += 1

        if run_cfg['analysis_type'] in [AnalysisType.ST_UNIMODAL, AnalysisType.ST_MULTIMODAL]:
            model = generate_st_model(run_cfg)
        else:
            model = None

        X_train_in = X_train_out[torch.tensor(inner_train_index)]
        X_val_in = X_train_out[torch.tensor(inner_val_index)]
        print("Inner Size is:", len(X_train_in), "/", len(X_val_in))
        print("Inner Positive classes:", sum([data.y.item() for data in X_train_in]),
              "/", sum([data.y.item() for data in X_val_in]))

        if run_cfg['analysis_type'] in [AnalysisType.ST_UNIMODAL, AnalysisType.ST_MULTIMODAL]:
            inner_fold_metrics = fit_st_model(out_fold_num=run_cfg['split_to_test'],
                                              in_fold_num=inner_loop_run,
                                              run_cfg=run_cfg,
                                              model=model,
                                              X_train_in=X_train_in,
                                              X_val_in=X_val_in)

            update_overall_metrics(overall_metrics, inner_fold_metrics)

    send_inner_loop_metrics_to_wandb(overall_metrics)

    # Calculating already on test set for quicker reporting
    test_out_loader = DataLoader(X_test_out, batch_size=run_cfg['batch_size'], shuffle=False, **kwargs_dataloader)
    print('Overall inner loop results:', overall_metrics)

    test_metrics = evaluate_classifier(model, test_out_loader, run_cfg['param_pooling'], run_cfg['device_run'])
    print(test_metrics)

    print('{:1d}-Final: {:.7f}, Auc: {:.4f}, Acc: {:.4f}, Sens: {:.4f}, Speci: {:.4f}'
          ''.format(outer_split_num, test_metrics['loss'], test_metrics['auc'], test_metrics['acc'],
                    test_metrics['sensitivity'], test_metrics['specificity']))
    send_global_results(test_metrics)
