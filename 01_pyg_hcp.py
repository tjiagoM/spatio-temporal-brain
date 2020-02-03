import argparse
import os
from sys import exit
import time

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import torch
import torch.nn.functional as F
from scipy.stats import stats
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import DataLoader

from datasets import HCPDataset
from model import SpatioTemporalModel
from utils import create_name_for_hcp_dataset, create_name_for_model, Normalisation, ConnType, ConvStrategy, \
    StratifiedGroupKFold, PoolingStrategy


def train_classifier(model, train_loader):
    model.train()
    loss_all = 0
    criterion = torch.nn.BCELoss()

    grads = {'final_l': [],
             'conv1d_1': []
             }
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        if POOLING == PoolingStrategy.DIFFPOOL:
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


def evaluate_classifier(loader, save_path_preds=None):
    model.eval()
    criterion = torch.nn.BCELoss()

    predictions = []
    labels = []
    test_error = 0

    for data in loader:
        with torch.no_grad():
            data = data.to(device)
            if POOLING == PoolingStrategy.DIFFPOOL:
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

    # TODO: Define accuracy at optimal AUC point
    # https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python

    roc_auc = roc_auc_score(labels, predictions)
    acc = accuracy_score(labels, pred_binary)
    f1 = f1_score(labels, pred_binary)

    return {'loss': test_error / len(loader.dataset),
            'auc': roc_auc,
            'acc': acc,
            'f1': f1
            }


def classifier_step(outer_split_no, inner_split_no, epoch, model, train_loader, val_loader):
    loss = train_classifier(model, train_loader)
    train_metrics = evaluate_classifier(train_loader)
    val_metrics = evaluate_classifier(val_loader)

    print(
        '{:1d}-{:1d}-Epoch: {:03d}, Loss: {:.7f} / {:.7f}, Auc: {:.4f} / {:.4f}, Acc: {:.4f} / {:.4f}, F1: {:.4f} / '
        '{:.4f} '.format(outer_split_no, inner_split_no, epoch, loss, val_metrics['loss'], train_metrics['auc'],
                         val_metrics['auc'],
                         train_metrics['acc'], val_metrics['acc'], train_metrics['f1'], val_metrics['f1']))

    return val_metrics


def merge_y_and_others(ys, sessions, directions):
    tmp = torch.cat([ys.long().view(-1, 1),
                     sessions.view(-1, 1),
                     directions.view(-1, 1)], dim=1)
    return LabelEncoder().fit_transform([str(l) for l in tmp.numpy()])


if __name__ == '__main__':

    import warnings

    warnings.filterwarnings("ignore")
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(1111)

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default="cuda")

    parser.add_argument("--fold_num", type=int)
    parser.add_argument("--target_var")
    parser.add_argument("--activation")
    parser.add_argument("--threshold", type=int)
    parser.add_argument("--num_nodes", type=int)
    parser.add_argument("--num_epochs", default=20, type=int)
    parser.add_argument("--batch_size", default=150, type=int)
    parser.add_argument("--add_gcn", type=bool, default=False)  # to make true just include flag with 1
    parser.add_argument("--add_gat", type=bool, default=False)  # to make true just include flag with 1
    parser.add_argument("--remove_disconnected_nodes", type=bool,
                        default=False)  # to make true just include flag with 1
    parser.add_argument("--conn_type", default="struct")
    parser.add_argument("--conv_strategy", default="entire")
    parser.add_argument("--pooling",
                        default="mean")  # 2) Try other pooling mechanisms CONCAT (only with fixed num_nodes across graphs),
    parser.add_argument("--channels_conv", type=int)
    parser.add_argument("--normalisation")

    args = parser.parse_args()

    # To check time execution
    start_time = time.time()

    # Device part
    device = torch.device(args.device)

    # Making a single variable for each argument
    N_EPOCHS = args.num_epochs
    TARGET_VAR = args.target_var
    ACTIVATION = args.activation
    THRESHOLD = args.threshold
    SPLIT_TO_TEST = args.fold_num
    ADD_GCN = args.add_gcn
    ADD_GAT = args.add_gat
    BATCH_SIZE = args.batch_size
    REMOVE_NODES = args.remove_disconnected_nodes
    NUM_NODES = args.num_nodes
    CONN_TYPE = ConnType(args.conn_type)
    CONV_STRATEGY = ConvStrategy(args.conv_strategy)
    POOLING = PoolingStrategy(args.pooling)
    CHANNELS_CONV = args.channels_conv
    NORMALISATION = Normalisation(args.normalisation)

    if NUM_NODES == 300 and CHANNELS_CONV > 1:
        BATCH_SIZE = int(BATCH_SIZE / 3)

    if TARGET_VAR not in ['gender']:
        print("Unrecognised target_var")
        exit(-1)
    else:
        print("Predicting", TARGET_VAR, N_EPOCHS, SPLIT_TO_TEST, ADD_GCN, ACTIVATION, THRESHOLD, ADD_GAT,
              BATCH_SIZE, REMOVE_NODES, NUM_NODES, CONN_TYPE, CONV_STRATEGY, POOLING, CHANNELS_CONV)

    #
    # Definition of general variables
    #
    name_dataset = create_name_for_hcp_dataset(num_nodes=NUM_NODES,
                                               target_var=TARGET_VAR,
                                               threshold=THRESHOLD,
                                               normalisation=NORMALISATION,
                                               connectivity_type=CONN_TYPE,
                                               disconnect_nodes=REMOVE_NODES)
    print("Going for", name_dataset)
    dataset = HCPDataset(root=name_dataset,
                         num_nodes=NUM_NODES,
                         target_var=TARGET_VAR,
                         threshold=THRESHOLD,
                         normalisation=NORMALISATION,
                         connectivity_type=CONN_TYPE,
                         disconnect_nodes=REMOVE_NODES)

    N_OUT_SPLITS = 5
    N_INNER_SPLITS = 5

    if TARGET_VAR == 'gender':
        # Stratification will occur with regards to both the sex and session day
        skf = StratifiedGroupKFold(n_splits=N_OUT_SPLITS, random_state=1111)
        merged_labels = merge_y_and_others(dataset.data.y,
                                           dataset.data.session,
                                           dataset.data.direction)
        skf_generator = skf.split(np.zeros((len(dataset), 1)),
                                  merged_labels,
                                  groups=dataset.data.hcp_id.tolist())
    else:
        print("Something wrong with target_var")
        exit(-1)

    #
    # Main outer-loop
    #
    outer_split_num = 0
    for train_index, test_index in skf_generator:
        outer_split_num += 1

        # Only run for the specific fold defined in the script arguments.
        if outer_split_num != SPLIT_TO_TEST:
            continue

        X_train_out = dataset[torch.tensor(train_index)]
        X_test_out = dataset[torch.tensor(test_index)]

        print("Size is:", len(X_train_out), "/", len(X_test_out))
        print("Positive classes:", sum(X_train_out.data.y.numpy()), "/", sum(X_test_out.data.y.numpy()))

        train_out_loader = DataLoader(X_train_out, batch_size=BATCH_SIZE, shuffle=True)
        test_out_loader = DataLoader(X_test_out, batch_size=BATCH_SIZE, shuffle=True)
        #
        # Main inner-loop (for now, not really an inner loop - just one train/val inside
        #
        param_grid = {'weight_decay': [0.005, 0.5, 0],
                      'lr': [1e-4, 1e-5, 1e-6],
                      'dropout': [0, 0.5, 0.7]
                      }
        # param_grid = {'weight_decay': [0],
        #              'lr': [0.05],
        #              'dropout': [0]
        #              }
        grid = ParameterGrid(param_grid)
        # best_metric = -100
        # best_params = None
        best_model_name_outer_fold_auc = None
        best_model_name_outer_fold_loss = None
        best_outer_metric_loss = 1000
        best_outer_metric_auc = -1000
        for params in grid:
            print("For ", params)

            if TARGET_VAR == 'gender':
                skf_inner = StratifiedGroupKFold(n_splits=N_INNER_SPLITS, random_state=1111)
                merged_labels_inner = merge_y_and_others(X_train_out.data.y,
                                                         X_train_out.data.session,
                                                         X_train_out.data.direction)
                skf_inner_generator = skf_inner.split(np.zeros((len(X_train_out), 1)),
                                                      merged_labels_inner,
                                                      groups=X_train_out.data.hcp_id.tolist())
                model_with_sigmoid = True
                metrics = ['acc', 'f1', 'auc', 'loss']

            # This for-cycle will only be executed once (for now)
            for inner_train_index, inner_val_index in skf_inner_generator:
                model = SpatioTemporalModel(num_time_length=1200,
                                            dropout_perc=params['dropout'],
                                            pooling=POOLING,
                                            channels_conv=CHANNELS_CONV,
                                            activation=ACTIVATION,
                                            conv_strategy=CONV_STRATEGY,
                                            add_gat=ADD_GAT,
                                            add_gcn=ADD_GCN,
                                            final_sigmoid=model_with_sigmoid,
                                            num_nodes=NUM_NODES
                                            ).to(device)
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print("Number of trainable params:", trainable_params)

                # Creating the various names for each metric
                model_names = {}
                for m in metrics:
                    model_names[m] = create_name_for_model(TARGET_VAR, model, params, outer_split_num, 0, N_EPOCHS,
                                                           THRESHOLD, BATCH_SIZE, REMOVE_NODES, NUM_NODES, CONN_TYPE,
                                                           NORMALISATION,
                                                           m)
                # If there is one of the metrics saved, then I assume this inner part was already calculated
                if os.path.isfile(model_names[metrics[0]]):
                    print("Saved model exists, thus skipping this search...")
                    break  # break because I'm in the "inner" fold, which is being done only once

                X_train_in = X_train_out[torch.tensor(inner_train_index)]
                X_val_in = X_train_out[torch.tensor(inner_val_index)]

                print("Inner Size is:", len(X_train_in), "/", len(X_val_in))
                print("Inner Positive classes:", sum(X_train_in.data.y.numpy()), "/", sum(X_val_in.data.y.numpy()))

                ###########
                ### DataLoaders
                train_in_loader = DataLoader(X_train_in, batch_size=BATCH_SIZE, shuffle=True)
                val_loader = DataLoader(X_val_in, batch_size=BATCH_SIZE, shuffle=True)

                optimizer = torch.optim.Adam(model.parameters(),
                                             lr=params['lr'],
                                             weight_decay=params['weight_decay'])

                best_metrics_fold = {}
                for m in metrics:
                    if m == 'loss':
                        best_metrics_fold[m] = 1000
                    else:
                        best_metrics_fold[m] = -1000
                for epoch in range(1, N_EPOCHS):
                    if TARGET_VAR == 'gender':
                        val_metrics = classifier_step(outer_split_num,
                                                      0,
                                                      epoch,
                                                      model,
                                                      train_in_loader,
                                                      val_loader)
                        if val_metrics['loss'] < best_metrics_fold['loss']:
                            best_metrics_fold['loss'] = val_metrics['loss']
                            torch.save(model, model_names['loss'])
                            if val_metrics['loss'] < best_outer_metric_loss:
                                best_outer_metric_loss = val_metrics['loss']
                                best_model_name_outer_fold_loss = model_names['loss']
                        if val_metrics['auc'] > best_metrics_fold['auc']:
                            best_metrics_fold['auc'] = val_metrics['auc']
                            torch.save(model, model_names['auc'])
                            if val_metrics['auc'] > best_outer_metric_auc:
                                best_outer_metric_auc = val_metrics['auc']
                                best_model_name_outer_fold_auc = model_names['auc']

                # End of inner-fold, put best val_metric in the array
                # if best_metric_fold > best_metric:
                #    best_metric = best_metric_fold
                #    print("New best val metric", best_metric)
                #    best_params = params
                #    torch.save(model, "logs/best_model_" + TARGET_VAR + "_" + str(ADD_GCN) + "_" + str(
                #        outer_split_num) + ".pth")
                break  # Just one inner "loop"
        if TARGET_VAR == 'gender':
            # After all parameters are searched, get best and train on that, evaluating on test set
            print("Best params if AUC: ", best_model_name_outer_fold_auc, "(", best_outer_metric_auc, ")")
            model = torch.load(best_model_name_outer_fold_auc)
            test_metrics = evaluate_classifier(test_out_loader,
                                               save_path_preds=best_model_name_outer_fold_auc.replace('logs/',
                                                                                                      '').replace(
                                                   '.pth', '.npy'))
            print('{:1d}-Final: {:.7f}, Auc: {:.4f}, Acc: {:.4f}, F1: {:.4f}'
                  ''.format(outer_split_num, test_metrics['loss'], test_metrics['auc'], test_metrics['acc'],
                            test_metrics['f1']))

            print("Best params if loss: ", best_model_name_outer_fold_loss, "(", best_outer_metric_loss, ")")
            model = torch.load(best_model_name_outer_fold_loss)
            test_metrics = evaluate_classifier(test_out_loader,
                                               save_path_preds=best_model_name_outer_fold_loss.replace('logs/',
                                                                                                       '').replace(
                                                   '.pth', '.npy'))
            print('{:1d}-Final: {:.7f}, Auc: {:.4f}, Acc: {:.4f}, F1: {:.4f}'
                  ''.format(outer_split_num, test_metrics['loss'], test_metrics['auc'], test_metrics['acc'],
                            test_metrics['f1']))

        # else:
        #    test_loss, test_r2, test_pear = evaluate_regressor(test_out_loader)

        #    print('{:1d}-Final: {:.7f}, R2: {:.4f}, Pear: {:.4f}'
        #          ''.format(outer_split_num, test_loss, test_r2, test_pear))

        # Conclusao para ja: definir apenas um validation set? .... (depois melhorar para nested-CV with fixed epochs
        # learned as average from the nested CV)

    print("--- %s seconds to execute this script---" % (time.time() - start_time))

