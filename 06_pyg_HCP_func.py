import argparse
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import torch
import torch.nn.functional as F
from scipy.stats import stats
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from torch_geometric.data import DataLoader

from datasets import HCPFunctDataset
from model import NetG
from utils import create_name_for_hcp_dataset, create_name_for_model

device = torch.device('cuda')


def train_classifier(model, train_loader):
    model.train()
    loss_all = 0
    criterion = torch.nn.BCELoss()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output_batch = model(data)
        loss = criterion(output_batch, data.y.unsqueeze(1))
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

    # len(train_loader) gives the number of batches
    # len(train_loader.dataset) gives the number of graphs

    # Returning a weighted average according to number of graphs
    return loss_all / len(train_loader.dataset)


def evaluate_classifier(loader):
    model.eval()
    criterion = torch.nn.BCELoss()

    predictions = []
    labels = []
    test_error = 0

    for data in loader:
        with torch.no_grad():
            data = data.to(device)

            output_batch = model(data).flatten()
            loss = criterion(output_batch, data.y)
            test_error += loss.item() * data.num_graphs

            pred = output_batch.detach().cpu().numpy()

            label = data.y.detach().cpu().numpy()
            predictions.append(pred)
            labels.append(label)

    predictions = np.hstack(predictions)
    labels = np.hstack(labels)

    pred_binary = np.where(predictions > 0.5, 1, 0)

    # TODO: Define accuracy at optimal AUC point
    # https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python

    roc_auc = roc_auc_score(labels, predictions)
    acc = accuracy_score(labels, pred_binary)
    f1 = f1_score(labels, pred_binary)

    return test_error / len(loader.dataset), roc_auc, acc, f1


def train_regressor(model, train_loader):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output_batch = model(data)
        loss = F.mse_loss(output_batch, data.y.unsqueeze(1))
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

    # len(train_loader) gives the number of batches
    # len(train_loader.dataset) gives the number of graphs

    # Returning a weighted average according to number of graphs
    return loss_all / len(train_loader.dataset)


# Tipo o YooChoose thing, e chamar este evaluate para o train e test, metendo
# para o cpu para calcular as coisas no final de cada epoch (incluindo o loss!)
# e tirar as metricas do train
def evaluate_regressor(loader):
    model.eval()

    predictions = []
    labels = []
    test_error = 0

    for data in loader:
        with torch.no_grad():
            data = data.to(device)

            output_batch = model(data).flatten()
            loss = F.mse_loss(output_batch, data.y)
            test_error += loss.item() * data.num_graphs

            pred = output_batch.detach().cpu().numpy()

            label = data.y.detach().cpu().numpy()
            predictions.append(pred)
            labels.append(label)

    predictions = np.hstack(predictions)
    labels = np.hstack(labels)

    return test_error / len(loader.dataset), r2_score(labels, predictions), stats.pearsonr(predictions, labels)[0]


def classifier_step(outer_split_no, inner_split_no, epoch, model, train_loader, val_loader):
    loss = train_classifier(model, train_loader)
    _, train_auc, train_acc, train_f1 = evaluate_classifier(train_loader)
    val_loss, val_auc, val_acc, val_f1 = evaluate_classifier(val_loader)

    print(
        '{:1d}-{:1d}-Epoch: {:03d}, Loss: {:.7f} / {:.7f}, Auc: {:.4f} / {:.4f}, Acc: {:.4f} / {:.4f}, F1: {:.4f} / {:.4f}'
        ''.format(outer_split_no, inner_split_no, epoch, loss, val_loss, train_auc, val_auc, train_acc, val_acc,
                  train_f1, val_f1))

    return val_loss, val_auc, val_acc, val_f1


def regression_step(outer_split_no, inner_split_no, epoch, model, train_loader, val_loader):
    loss = train_regressor(model, train_loader)
    _, train_r2, train_pear = evaluate_regressor(train_loader)
    val_loss, val_r2, val_pear = evaluate_regressor(val_loader)

    print('{:1d}-{:1d}-Epoch: {:03d}, Loss: {:.7f} / {:.7f}, R2: {:.4f} / {:.4f}, Pear: {:.4f} / {:.4f}'
          ''.format(outer_split_no, inner_split_no, epoch, loss, val_loss, train_r2, val_r2, train_pear, val_pear))

    return val_r2


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--fold_num", type=int)
    parser.add_argument("--target_var")
    parser.add_argument("--activation")
    parser.add_argument("--threshold", type=int)
    parser.add_argument("--num_nodes", type=int)
    parser.add_argument("--num_epochs", default=75, type=int)
    parser.add_argument("--batch_size", default=150, type=int)
    parser.add_argument("--add_gcn", type=bool, default=False)  # to make true just include flag with 1
    parser.add_argument("--add_gat", type=bool, default=False)  # to make true just include flag with 1
    parser.add_argument("--remove_disconnected_nodes", type=bool,
                        default=False)  # to make true just include flag with 1
    parser.add_argument("--conn_type", default="fmri")
    parser.add_argument("--conv_strategy", default="entire")
    parser.add_argument("--pooling",
                        default="mean")  # 2) Try other pooling mechanisms CONCAT (only with fixed num_nodes across graphs), dense_diff_pool, SAGPooling (attention!), TopKPooling
    parser.add_argument("--channels_conv", type=int)

    args = parser.parse_args()
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
    CONN_TYPE = args.conn_type
    CONV_STRATEGY = args.conv_strategy
    POOLING = args.pooling
    CHANNELS_CONV = args.channels_conv

    if NUM_NODES == 300 and CHANNELS_CONV > 1:
        BATCH_SIZE = int(BATCH_SIZE / 3)

    # 1) Start with simple architectures (maybe not a big combinations) to get preliminary results
    # 3) Try other graph creations.
    # 4) Multimodal bringing the structural data?

    if TARGET_VAR not in ['gender', 'intelligence']:
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
                                               connectivity_type=CONN_TYPE,
                                               disconnect_nodes=REMOVE_NODES)
    print("Going for", name_dataset)
    dataset = HCPFunctDataset(root=name_dataset,
                              num_nodes=NUM_NODES,
                              target_var=TARGET_VAR,
                              threshold=THRESHOLD,
                              connectivity_type=CONN_TYPE,
                              disconnect_nodes=REMOVE_NODES)

    N_OUT_SPLITS = 10
    N_INNER_SPLITS = 5

    if TARGET_VAR == 'gender':
        skf = StratifiedKFold(n_splits=N_OUT_SPLITS, shuffle=True, random_state=0)
        skf_generator = skf.split(np.zeros((len(dataset), 1)), dataset.data.y.numpy())
    elif TARGET_VAR == 'intelligence':
        scores = pd.read_csv('confounds.csv').set_index('Subject')
        scores = scores[['g_efa', 'Handedness', 'Age_in_Yrs', 'FS_BrainSeg_Vol', 'Gender', 'fMRI_3T_ReconVrs']]
        skf = KFold(n_splits=N_OUT_SPLITS, shuffle=True, random_state=0)
        skf_generator = skf.split(np.zeros((len(dataset), 1)))
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

        # If predicting intelligence, ncessary to update labels with residuals
        # TODO: Make residual corrections inside inner-loop
        if TARGET_VAR == 'intelligence':
            ids = [elem.hcp_id.item() for elem in X_train_out]

            scores_filt = scores.loc[ids, :]
            regress_model = smf.ols(
                formula='g_efa ~ Handedness + Age_in_Yrs + FS_BrainSeg_Vol + C(Gender) + C(fMRI_3T_ReconVrs)',
                data=scores_filt).fit()

            for elem in X_train_out:
                elem.y[0] = regress_model.resid.loc[elem.hcp_id.item()]

            ids = [elem.hcp_id.item() for elem in X_test_out]
            scores_filt = scores.loc[ids, :]

            scores_test_residuals = scores_filt['g_efa'] - regress_model.predict(scores_filt)

            for elem in X_test_out:
                elem.y[0] = scores_test_residuals.loc[elem.hcp_id.item()]

        train_out_loader = DataLoader(X_train_out, batch_size=BATCH_SIZE, shuffle=True)
        test_out_loader = DataLoader(X_test_out, batch_size=BATCH_SIZE, shuffle=True)
        #
        # Main inner-loop (for now, not really an inner loop - just one train/val inside
        #
        param_grid = {'weight_decay': [0.0005, 0.005, 0.05, 0.5, 0],
                      'lr': [0.0005, 0.005, 0.05, 0.5],
                      'dropout': [0, 0.3, 0.5, 0.7, 0.9]
                      }
        grid = ParameterGrid(param_grid)
        # best_metric = -100
        # best_params = None
        for params in grid:
            print("For ", params)

            if TARGET_VAR == 'gender':
                skf_inner = StratifiedKFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=0)
                skf_inner_generator = skf_inner.split(np.zeros((len(X_train_out), 1)), X_train_out.data.y.numpy())
                model_with_sigmoid = True
                metrics = ['acc', 'f1', 'auc', 'loss']
            else:
                skf_inner = KFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=0)
                skf_inner_generator = skf_inner.split(np.zeros((len(X_train_out), 1)))
                model_with_sigmoid = False
                metrics = ['r2', 'pears', 'loss']

            # This for-cycle will only be executed once (for now)
            for inner_train_index, inner_val_index in skf_inner_generator:
                model = NetG(num_time_length=4800,
                             dropout_perc=params['dropout'],
                             pooling=POOLING,
                             channels_conv=CHANNELS_CONV,
                             activation=ACTIVATION,
                             conv_strategy=CONV_STRATEGY,
                             add_gat=ADD_GAT,
                             add_gcn=ADD_GCN,
                             final_sigmoid=model_with_sigmoid
                             ).to(device)

                # Creating the various names for each metric
                model_names = {}
                for m in metrics:
                    model_names[m] = create_name_for_model(TARGET_VAR, model, params, outer_split_num, 0,
                                                           N_EPOCHS, THRESHOLD, BATCH_SIZE, REMOVE_NODES, NUM_NODES,
                                                           CONN_TYPE,
                                                           m)
                # If there is one of the metrics saved, then I assume this inner part was already calculated
                if os.path.isfile(model_names[metrics[0]]):
                    print("Saved model exists, thus skipping this search...")
                    break  # break because I'm in the "inner" fold, which is being done only once

                X_train_in = X_train_out[torch.tensor(inner_train_index)]
                X_val_in = X_train_out[torch.tensor(inner_val_index)]

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
                        # TODO: Make this a dict to be easier to compare
                        val_loss, val_auc, val_acc, val_f1 = classifier_step(outer_split_num,
                                                                             0,
                                                                             epoch,
                                                                             model,
                                                                             train_in_loader,
                                                                             val_loader)
                        if val_loss < best_metrics_fold['loss']:
                            best_metrics_fold['loss'] = val_loss
                            torch.save(model, model_names['loss'])
                        if val_acc > best_metrics_fold['acc']:
                            best_metrics_fold['acc'] = val_acc
                            torch.save(model, model_names['acc'])
                        if val_auc > best_metrics_fold['auc']:
                            best_metrics_fold['auc'] = val_auc
                            torch.save(model, model_names['auc'])
                        if val_f1 > best_metrics_fold['f1']:
                            best_metrics_fold['f1'] = val_f1
                            torch.save(model, model_names['f1'])

                    elif TARGET_VAR == 'intelligence':
                        val_metric = regression_step(outer_split_num,
                                                     0,
                                                     epoch,
                                                     model,
                                                     train_in_loader,
                                                     val_loader)

                # End of inner-fold, put best val_metric in the array
                # if best_metric_fold > best_metric:
                #    best_metric = best_metric_fold
                #    print("New best val metric", best_metric)
                #    best_params = params
                #    torch.save(model, "logs/best_model_" + TARGET_VAR + "_" + str(ADD_GCN) + "_" + str(
                #        outer_split_num) + ".pth")
                break  # Just one inner "loop"

        # After all parameters are searched, get best and train on that, evaluating on test set
        # print("Best params: ", best_params)
        # model = torch.load("logs/best_model_" + TARGET_VAR + "_" + str(ADD_GCN) + "_" + str(outer_split_num) + ".pth")

        # if TARGET_VAR == 'gender':
        #    test_loss, test_auc, test_acc = evaluate_classifier(test_out_loader)

        #    print('{:1d}-Final: {:.7f}, Auc: {:.4f}, Acc: {:.4f}'
        #          ''.format(outer_split_num, test_loss, test_auc, test_acc))
        # else:
        #    test_loss, test_r2, test_pear = evaluate_regressor(test_out_loader)

        #    print('{:1d}-Final: {:.7f}, R2: {:.4f}, Pear: {:.4f}'
        #          ''.format(outer_split_num, test_loss, test_r2, test_pear))

        # Conclusao para ja: definir apenas um validation set? .... (depois melhorar para nested-CV with fixed epochs
        # learned as average from the nested CV)