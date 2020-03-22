import argparse
import copy
import datetime
import pickle
import random
import time
from collections import deque
from sys import exit

import numpy as np
import torch
torch.multiprocessing.set_start_method('spawn', force=True)
import wandb
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from torch_geometric.data import DataLoader
from xgboost import XGBClassifier

from datasets import BrainDataset, create_hcp_correlation_vals, create_ukb_corrs_flatten
from model import SpatioTemporalModel
from utils import create_name_for_brain_dataset, create_name_for_model, Normalisation, ConnType, ConvStrategy, \
    StratifiedGroupKFold, PoolingStrategy, AnalysisType, merge_y_and_others, EncodingStrategy, create_best_encoder_name, \
    get_best_model_paths
from wandb_utils import SWEEP_GENERAL

def train_classifier(model, train_loader, optimizer, pooling_mechanism):
    model.train()
    loss_all = 0
    criterion = torch.nn.BCELoss()

    grads = {'final_l': [],
             'conv1d_1': []
             }
    for data in train_loader:
        data = data.to(wandb.config.device)
        optimizer.zero_grad()
        if pooling_mechanism == PoolingStrategy.DIFFPOOL:
            output_batch, link_loss, ent_loss = model(data)
            loss = criterion(output_batch, data.y.unsqueeze(1)) + link_loss + ent_loss
        else:
            output_batch = model(data)
            loss = criterion(output_batch, data.y.unsqueeze(1))

        loss.backward()

        grads['final_l'].extend(model.final_linear.weight.grad.flatten().cpu().tolist())
        grads['conv1d_1'].extend(model.final_linear.weight.grad.flatten().cpu().tolist())

        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    print("GRAD", np.mean(grads['final_l']), np.std(grads['final_l']))
    # len(train_loader) gives the number of batches
    # len(train_loader.dataset) gives the number of graphs

    # Returning a weighted average according to number of graphs
    return loss_all / len(train_loader.dataset)


def return_metrics(labels, pred_binary, pred_prob, loss_value=None):
    roc_auc = roc_auc_score(labels, pred_prob)
    acc = accuracy_score(labels, pred_binary)
    f1 = f1_score(labels, pred_binary, zero_division=0)
    report = classification_report(labels, pred_binary, output_dict=True, zero_division=0)
    sens = report['1.0']['recall']
    spec = report['0.0']['recall']

    return {'loss': loss_value,
            'auc': roc_auc,
            'acc': acc,
            'f1': f1,
            'sensitivity': sens,
            'specificity': spec
            }


def evaluate_classifier(model, loader, pooling_mechanism, device, save_path_preds=None):
    model.eval()
    criterion = torch.nn.BCELoss()

    predictions = []
    labels = []
    test_error = 0

    for data in loader:
        with torch.no_grad():
            data = data.to(device)
            if pooling_mechanism == PoolingStrategy.DIFFPOOL:
                output_batch, link_loss, ent_loss = model(data)
                output_batch = output_batch.flatten()
                loss = criterion(output_batch, data.y) + link_loss + ent_loss
            else:
                output_batch = model(data)
                output_batch = output_batch.flatten()
                loss = criterion(output_batch, data.y)

            test_error += loss.item() * data.num_graphs

            pred = output_batch.detach().cpu().numpy()

            label = data.y.detach().cpu().numpy()
            predictions.append(pred)
            labels.append(label)
    predictions = np.hstack(predictions)
    labels = np.hstack(labels)

    if save_path_preds is not None:
        np.save('results/labels_' + save_path_preds, labels)
        np.save('results/predictions_' + save_path_preds, predictions)

    pred_binary = np.where(predictions > 0.5, 1, 0)

    return return_metrics(labels, pred_binary, predictions, loss_value=test_error / len(loader.dataset))


def classifier_step(outer_split_no, inner_split_no, epoch, model, train_loader, val_loader, optimizer,
                    pooling_mechanism, device):
    loss = train_classifier(model, train_loader, optimizer, pooling_mechanism)
    train_metrics = evaluate_classifier(model, train_loader, pooling_mechanism, device)
    val_metrics = evaluate_classifier(model, val_loader, pooling_mechanism, device)

    print('{:1d}-{:1d}-Epoch: {:03d}, Loss: {:.7f} / {:.7f}, Auc: {:.4f} / {:.4f}, Acc: {:.4f} / {:.4f}, F1: {:.4f} /'
          ' {:.4f} '.format(outer_split_no, inner_split_no, epoch, loss, val_metrics['loss'],
                            train_metrics['auc'], val_metrics['auc'],
                            train_metrics['acc'], val_metrics['acc'],
                            train_metrics['f1'], val_metrics['f1']))
    wandb.log({
        'train_loss': loss, 'val_loss': val_metrics['loss'],
        'train_auc': train_metrics['auc'], 'val_auc': val_metrics['auc'],
        'train_acc': train_metrics['acc'], 'val_acc': val_metrics['acc'],
        'train_sens': train_metrics['sensitivity'], 'val_sens': val_metrics['sensitivity'],
        'train_spec': train_metrics['specificity'], 'val_spec': val_metrics['specificity']
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


def create_fold_generator(dataset, num_nodes, num_splits):
    # UK Biobank
    if num_nodes == 376:
        skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=1111)
        skf_generator = skf.split(np.zeros((len(dataset), 1)),
                                  np.array([data.y.item() for data in dataset]))
    else:
        # Stratification will occur with regards to both the sex and session day
        skf = StratifiedGroupKFold(n_splits=num_splits, random_state=1111)
        merged_labels = merge_y_and_others(torch.cat([data.y for data in dataset], dim=0),
                                           torch.cat([data.index for data in dataset], dim=0))
        skf_generator = skf.split(np.zeros((len(dataset), 1)),
                                  merged_labels,
                                  groups=[data.hcp_id.item() for data in dataset])

    return skf_generator


def main_loop():
    wandb.init()
    config = wandb.config
    print('Config file from wandb:', config)

    # import warnings
    # warnings.filterwarnings("ignore")
    torch.manual_seed(1)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(1111)
    random.seed(1111)
    torch.cuda.manual_seed_all(1111)

    # To check time execution
    start_time = time.time()

    # Device part
    device = torch.device(config.device)
    # Needed because of forked problems with DataLoader
    #kwargs = {'num_workers': 1, 'pin_memory': True} if config.device.startswith('cuda') else {}
    #kwargs = {}
    #print('PP', kwargs)

    # Making a single variable for each argument
    num_epochs = config.num_epochs
    target_var = config.target_var
    param_activation = config.activation
    param_threshold = config.threshold
    split_to_test = config.fold_num
    param_num_gnn_layers = config.num_gnn_layers
    param_add_gcn = True if config.gnn_type == 'gcn' and param_num_gnn_layers > 0 else False
    param_add_gat = True if config.gnn_type == 'gat' and param_num_gnn_layers > 0 else False
    batch_size = config.batch_size
    param_remove_nodes = config.remove_disconnected_nodes
    num_nodes = config.num_nodes
    param_conn_type = ConnType(config.conn_type)
    param_conv_strategy = ConvStrategy(config.conv_strategy)
    param_pooling = PoolingStrategy(config.pooling)
    param_channels_conv = config.channels_conv
    param_normalisation = Normalisation(config.normalisation)
    analysis_type = AnalysisType(config.analysis_type)
    time_length = config.time_length
    ts_spit_num = int(4800 / time_length)
    param_encoding_strategy = EncodingStrategy(config.encoding_strategy)
    early_stop_steps = config.early_stop_steps
    param_dropout = config.dropout
    param_weight_decay = config.weight_decay
    param_lr = config.lr

    model_with_sigmoid = True

    N_OUT_SPLITS = 5
    N_INNER_SPLITS = 5

    if analysis_type != AnalysisType.SPATIOTEMOPRAL:
        print("Not yet ready for flatten predictions!")
        exit(-1)

    # if param_conv_strategy != ConvStrategy.TCN_ENTIRE:
    #    print("Setting to deterministic runs")
    #    torch.backends.cudnn.deterministic = True
    # else:
    #    print("This run will not be deterministic")
    print("This run will not be deterministic")
    if target_var not in ['gender']:
        print("Unrecognised target_var")
        exit(-1)
    else:
        print("Predicting", target_var, num_epochs, split_to_test, param_add_gcn, param_activation, param_threshold, param_add_gat,
              batch_size, param_remove_nodes, num_nodes, param_conn_type, param_conv_strategy, param_pooling, param_channels_conv, time_length)

    #
    # Definition of general variables
    #
    name_dataset = create_name_for_brain_dataset(num_nodes=num_nodes,
                                                 time_length=time_length,
                                                 target_var=target_var,
                                                 threshold=param_threshold,
                                                 normalisation=param_normalisation,
                                                 connectivity_type=param_conn_type,
                                                 disconnect_nodes=param_remove_nodes)
    print("Going for", name_dataset)
    dataset = BrainDataset(root=name_dataset,
                           time_length=time_length,
                           num_nodes=num_nodes,
                           target_var=target_var,
                           threshold=param_threshold,
                           normalisation=param_normalisation,
                           connectivity_type=param_conn_type,
                           disconnect_nodes=param_remove_nodes)
    if analysis_type == AnalysisType.FLATTEN_CORRS:
        if num_nodes == 376:
            flatten_correlations = create_ukb_corrs_flatten()
        else:
            flatten_correlations = create_hcp_correlation_vals(num_nodes, ts_split_num=ts_spit_num)
    elif analysis_type == AnalysisType.FLATTEN_CORRS_THRESHOLD:
        flatten_correlations = create_hcp_correlation_vals(num_nodes, ts_split_num=ts_spit_num,
                                                           binarise=True, threshold=param_threshold)

    skf_outer_generator = create_fold_generator(dataset, num_nodes, N_OUT_SPLITS)
    #################
    # Main outer-loop
    #################
    outer_split_num = 0
    for train_index, test_index in skf_outer_generator:
        outer_split_num += 1

        # Only run for the specific fold defined in the script arguments.
        if outer_split_num != split_to_test:
            continue

        X_train_out = dataset[torch.tensor(train_index)]
        X_test_out = dataset[torch.tensor(test_index)]

        print("Size is:", len(X_train_out), "/", len(X_test_out))
        print("Positive classes:", sum([data.y.item() for data in X_train_out]),
              "/", sum([data.y.item() for data in X_test_out]))

        train_out_loader = DataLoader(X_train_out, batch_size=batch_size, shuffle=True)
        test_out_loader = DataLoader(X_test_out, batch_size=batch_size, shuffle=False)

        #elif analysis_type == AnalysisType.FLATTEN_CORRS or analysis_type == AnalysisType.FLATTEN_CORRS_THRESHOLD:
        #    param_grid = {
        #        'min_child_weight': [1],  # , 5],
        #        'gamma': [0.0, 1, 5],
        #        'subsample': [0.6, 1.0],
        #        'colsample_bytree': [0.6, 1.0],
        #        'max_depth': [3],  # , 5],
        #        'n_estimators': [100, 500]
        #    }

        #grid = ParameterGrid(param_grid)
        # best_metric = -100
        # best_params = None
        #best_model_name_outer_fold_auc = None
        #best_model_name_outer_fold_loss = None
        #best_outer_metric_loss = 1000
        #best_outer_metric_auc = -1000
        #for params in grid:
        #    print("For ", params)

        skf_inner_generator = create_fold_generator(X_train_out, num_nodes, N_INNER_SPLITS)
        # Metrics are loss (for st model) and auc (for xgboost)
        metrics = ['auc', 'loss']

        # Getting the best values from previous run
        best_loss_path, best_name_path = get_best_model_paths(analysis_type.value, num_nodes, time_length, target_var,
                             split_to_test, param_conn_type.value, num_epochs, batch_size)
        with open(best_name_path, 'r') as f:
            best_model_name_outer_fold_loss = f.read()
        best_outer_metric_loss = np.load(best_loss_path)[0]
        print('Previous best loss value/name:', best_outer_metric_loss, best_model_name_outer_fold_loss)
        #################
        # Main inner-loop (for now, not really an inner loop - just one train/val inside
        #################
        for inner_train_index, inner_val_index in skf_inner_generator:
            if analysis_type == AnalysisType.SPATIOTEMOPRAL:
                if param_encoding_strategy != EncodingStrategy.NONE:
                    if param_encoding_strategy == EncodingStrategy.AE3layers:
                        from encoders import AE  # Necessary to torch.load
                    elif param_encoding_strategy == EncodingStrategy.VAE3layers:
                        from encoders import VAE  # Necessary to torch.load
                    encoding_model = torch.load(create_best_encoder_name(ts_length=time_length,
                                                                         outer_split_num=outer_split_num,
                                                                         encoder_name=param_encoding_strategy.value))
                else:
                    encoding_model = None
                model = SpatioTemporalModel(num_time_length=time_length,
                                            dropout_perc=param_dropout,
                                            pooling=param_pooling,
                                            channels_conv=param_channels_conv,
                                            activation=param_activation,
                                            conv_strategy=param_conv_strategy,
                                            add_gat=param_add_gat,
                                            add_gcn=param_add_gcn,
                                            final_sigmoid=model_with_sigmoid,
                                            num_nodes=num_nodes,
                                            num_gnn_layers=param_num_gnn_layers,
                                            encoding_model=encoding_model
                                            ).to(device)
                wandb.watch(model, log=None)
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print("Number of trainable params:", trainable_params)
            #elif analysis_type == AnalysisType.FLATTEN_CORRS or analysis_type == AnalysisType.FLATTEN_CORRS_THRESHOLD:
            #    model = XGBClassifier(n_jobs=-1, seed=1111, random_state=1111, **params)

            # Creating the various names for each metric
            model_names = {}
            for m in metrics:
                model_names[m] = create_name_for_model(target_var, model, outer_split_num, 0, num_epochs,
                                                       param_threshold, batch_size, param_remove_nodes, num_nodes, param_conn_type,
                                                       param_normalisation, analysis_type,
                                                       m, param_lr, param_weight_decay)

            X_train_in = X_train_out[torch.tensor(inner_train_index)]
            X_val_in = X_train_out[torch.tensor(inner_val_index)]

            print("Inner Size is:", len(X_train_in), "/", len(X_val_in))
            print("Inner Positive classes:", sum([data.y.item() for data in X_train_in]),
                  "/", sum([data.y.item() for data in X_val_in]))

            if analysis_type == AnalysisType.FLATTEN_CORRS or analysis_type == AnalysisType.FLATTEN_CORRS_THRESHOLD:
                X_train_in_array, y_train_in_array = get_array_data(flatten_correlations, X_train_in,
                                                                    num_nodes=num_nodes)
                X_val_in_array, y_val_in_array = get_array_data(flatten_correlations, X_val_in, num_nodes=num_nodes)

                model.fit(X_train_in_array, y_train_in_array)
                y_pred = model.predict(X_val_in_array)

                val_metrics = return_metrics(y_val_in_array, y_pred, y_pred)
                print(val_metrics)
                if val_metrics['auc'] > best_outer_metric_auc:
                    pickle.dump(model, open(model_names['auc'], "wb"))
                    best_outer_metric_auc = val_metrics['auc']
                    best_model_name_outer_fold_auc = model_names['auc']
                break

            ###########
            ### DataLoaders
            train_in_loader = DataLoader(X_train_in, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(X_val_in, batch_size=batch_size, shuffle=False)

            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=param_lr,
                                         weight_decay=param_weight_decay)

            best_metrics_fold = {}
            for m in metrics:
                if m == 'loss':
                    best_metrics_fold[m] = 9999
                else:
                    best_metrics_fold[m] = -9999
            # Only for loss
            last_losses_val = deque([9999 for _ in range(early_stop_steps)], maxlen=early_stop_steps)

            for epoch in range(num_epochs):
                if target_var == 'gender':
                    val_metrics = classifier_step(outer_split_num,
                                                  0,
                                                  epoch,
                                                  model,
                                                  train_in_loader,
                                                  val_loader,
                                                  optimizer,
                                                  param_pooling,
                                                  device)
                    if sum([val_metrics['loss'] > loss for loss in last_losses_val]) == early_stop_steps:
                        print("EARLY STOPPING IT")
                        break
                    last_losses_val.append(val_metrics['loss'])

                    if val_metrics['loss'] < best_metrics_fold['loss']:
                        best_metrics_fold['loss'] = val_metrics['loss']
                        torch.save(model, model_names['loss'])
                        if val_metrics['loss'] < best_outer_metric_loss:
                            best_outer_metric_loss = val_metrics['loss']
                            best_model_name_outer_fold_loss = model_names['loss']
                            wandb.save('best_model.h5')

            wandb.run.summary["best_val_loss"] = best_metrics_fold['loss']


            np.save(file=best_loss_path, arr=np.array([best_outer_metric_loss], dtype=float))
            with open(best_name_path, 'w') as f:
                f.write(best_model_name_outer_fold_loss)
            break  # Just one inner "loop"


    total_seconds = time.time() - start_time
    total_time = str(datetime.timedelta(seconds=total_seconds))
    print(f'--- {total_seconds} seconds to execute this script ({total_time})---')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fold_num", type=int)
    parser.add_argument("--target_var")
    parser.add_argument("--num_nodes", default=50, type=int)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=400, type=int)
    parser.add_argument("--conn_type", default="fmri")
    parser.add_argument("--analysis_type", default='spatiotemporal')
    parser.add_argument("--time_length", type=int)
    parser.add_argument("--early_stop_steps", default=30, type=int)

    parser.add_argument("--hyperparam_count", type=int, default=100)

    args = parser.parse_args()

    sweep_config = copy.deepcopy(SWEEP_GENERAL)
    sweep_config['name'] = f'{args.analysis_type[:3]}_{args.num_nodes}_{args.target_var[:2]}_{args.fold_num}'
    sweep_config['parameters']['device']['value'] = args.device
    sweep_config['parameters']['fold_num']['value'] = args.fold_num
    sweep_config['parameters']['target_var']['value'] = args.target_var
    sweep_config['parameters']['num_nodes']['value'] = args.num_nodes
    sweep_config['parameters']['num_epochs']['value'] = args.num_epochs
    sweep_config['parameters']['batch_size']['value'] = args.batch_size
    sweep_config['parameters']['conn_type']['value'] = args.conn_type
    sweep_config['parameters']['analysis_type']['value'] = args.analysis_type
    sweep_config['parameters']['time_length']['value'] = args.time_length
    sweep_config['parameters']['early_stop_steps']['value'] = args.early_stop_steps

    # First time, so it needs to clear the previous logs
    _, _ = get_best_model_paths(args.analysis_type, args.num_nodes,
                                args.time_length, args.target_var,
                                args.fold_num, args.conn_type,
                                args.num_epochs, args.batch_size,
                                first_time=True)

    sweep_id = wandb.sweep(sweep_config, entity='tjiagom', project="spatio-temporal-brain")

    wandb.agent(sweep_id, function=main_loop, count=args.hyperparam_count)

    #############
    # Reporting final metrics on test set
    #############
    print("Sweep done, reporting metrics on final test set")
    #print(dill.license()) # jsut for refactor does not delete import
    # Getting the best results from the sweep
    best_sweep_loss_path, best_sweep_name_path = get_best_model_paths(args.analysis_type, args.num_nodes,
                                                                      args.time_length, args.target_var,
                                                                      args.fold_num, args.conn_type,
                                                                      args.num_epochs, args.batch_size)
    with open(best_sweep_name_path, 'r') as f:
        best_model_name_sweep = f.read()
    best_loss_sweep = np.load(best_sweep_loss_path)[0]

    best_pooling = best_model_name_sweep.split('P_')[1].split('__')[0]
    best_threshold = int(best_model_name_sweep.split('THRE_')[1].split('_')[0])

    param_pooling = PoolingStrategy(best_pooling)
    analysis_type = AnalysisType(args.analysis_type)
    param_conn_type = ConnType(args.conn_type)
    param_normalisation = Normalisation(sweep_config['parameters']['normalisation']['values'][0])

    # This is a bit inefficient, but because of wandb, global variables seem to behave strangely
    N_OUT_SPLITS = 5
    name_dataset = create_name_for_brain_dataset(num_nodes=args.num_nodes,
                                                 time_length=args.time_length,
                                                 target_var=args.target_var,
                                                 threshold=best_threshold,
                                                 normalisation=param_normalisation,
                                                 connectivity_type=param_conn_type,
                                                 disconnect_nodes=sweep_config['parameters']['remove_disconnected_nodes']['value'])
    print("Going for", name_dataset)
    dataset = BrainDataset(root=name_dataset,
                           time_length=args.time_length,
                           num_nodes=args.num_nodes,
                           target_var=args.target_var,
                           threshold=best_threshold,
                           normalisation=param_normalisation,
                           connectivity_type=param_conn_type,
                           disconnect_nodes=sweep_config['parameters']['remove_disconnected_nodes']['value'])

    skf_outer_generator = create_fold_generator(dataset, args.num_nodes, N_OUT_SPLITS)
    outer_split_num = 0
    for train_index, test_index in skf_outer_generator:
        outer_split_num += 1

        # Only run for the specific fold defined in the script arguments.
        if outer_split_num != args.fold_num:
            continue

        X_test_out = dataset[torch.tensor(test_index)]
        test_out_loader = DataLoader(X_test_out, batch_size=args.batch_size, shuffle=False)
        break



    device = torch.device(args.device)
    if args.analysis_type == AnalysisType.SPATIOTEMOPRAL.value:
        print("Best val loss: ", best_loss_sweep, "(", best_model_name_sweep, ")")
        model = torch.load(best_model_name_sweep)
        saving_path = best_model_name_sweep.replace('logs/', '').replace('.pth', '.npy')
        test_metrics = evaluate_classifier(model, test_out_loader, param_pooling, device, save_path_preds=saving_path)

        print('{:1d}-Final: {:.7f}, Auc: {:.4f}, Acc: {:.4f}, Sens: {:.4f}, Speci: {:.4f}'
              ''.format(outer_split_num, test_metrics['loss'], test_metrics['auc'], test_metrics['acc'],
                        test_metrics['sensitivity'], test_metrics['specificity']))
    '''
    elif analysis_type == AnalysisType.FLATTEN_CORRS or analysis_type == AnalysisType.FLATTEN_CORRS_THRESHOLD:
        print(f'Best params if auc: {best_model_name_outer_fold_auc} ( {best_outer_metric_auc} )')
        model = pickle.load(open(best_model_name_outer_fold_auc, "rb"))

        X_test_array, y_test_array = get_array_data(flatten_correlations, X_test_out, num_nodes=num_nodes)
        y_pred = model.predict(X_test_array)
        test_metrics = return_metrics(y_test_array, y_pred, y_pred)
        print('{:1d}-Final: Auc: {:.4f}, Acc: {:.4f}, Sens: {:.4f}, Speci: {:.4f}'
              ''.format(outer_split_num, test_metrics['auc'], test_metrics['acc'],
                        test_metrics['sensitivity'], test_metrics['specificity']))

        save_path_preds = best_model_name_outer_fold_auc.replace('logs/', '').replace('.pkl', '.npy')

        np.save('results/labels_' + save_path_preds, y_test_array)
        np.save('results/predictions_' + save_path_preds, y_pred)
        '''
