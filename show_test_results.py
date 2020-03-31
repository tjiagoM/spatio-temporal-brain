import argparse
from collections import defaultdict

import numpy as np
import torch
import wandb
from torch_geometric.data import DataLoader

from datasets import BrainDataset
from main_loop import evaluate_classifier, create_fold_generator
from utils import create_name_for_brain_dataset, Normalisation, ConnType, PoolingStrategy, AnalysisType, \
    get_best_model_paths


def report_on_fold(fold_num, args):
    # Getting the best results from the sweep
    best_sweep_loss_path, best_sweep_name_path = get_best_model_paths(args.analysis_type, args.num_nodes,
                                                                      args.time_length, args.target_var,
                                                                      fold_num, args.conn_type,
                                                                      args.num_epochs, args.sweep_type)
    with open(best_sweep_name_path, 'r') as f:
        best_model_name_sweep = f.read()
    best_loss_sweep = np.load(best_sweep_loss_path)[0]

    best_pooling = best_model_name_sweep.split('P_')[1].split('__')[0]
    best_threshold = int(best_model_name_sweep.split('THRE_')[1].split('_')[0])

    param_pooling = PoolingStrategy(best_pooling)
    analysis_type = AnalysisType(args.analysis_type)
    param_conn_type = ConnType(args.conn_type)
    param_normalisation = Normalisation(args.normalisation)

    # This is a bit inefficient, but because of wandb, global variables seem to behave strangely
    N_OUT_SPLITS = 5
    name_dataset = create_name_for_brain_dataset(num_nodes=args.num_nodes,
                                                 time_length=args.time_length,
                                                 target_var=args.target_var,
                                                 threshold=best_threshold,
                                                 normalisation=param_normalisation,
                                                 connectivity_type=param_conn_type,
                                                 disconnect_nodes=args.remove_disconnected_nodes)
    #print("Going for", name_dataset)
    dataset = BrainDataset(root=name_dataset,
                           time_length=args.time_length,
                           num_nodes=args.num_nodes,
                           target_var=args.target_var,
                           threshold=best_threshold,
                           normalisation=param_normalisation,
                           connectivity_type=param_conn_type,
                           disconnect_nodes=args.remove_disconnected_nodes)

    skf_outer_generator = create_fold_generator(dataset, args.num_nodes, N_OUT_SPLITS)
    outer_split_num = 0
    for train_index, test_index in skf_outer_generator:
        outer_split_num += 1

        # Only run for the specific fold defined in the script arguments.
        if outer_split_num != fold_num:
            continue

        X_test_out = dataset[torch.tensor(test_index)]
        test_out_loader = DataLoader(X_test_out, batch_size=args.batch_size, shuffle=False)
        break

    device = torch.device(args.device)
    if args.analysis_type == AnalysisType.SPATIOTEMOPRAL.value:
        print("Best val loss: ", best_loss_sweep, "(", best_model_name_sweep, ")")
        model = torch.load(best_model_name_sweep, map_location=device)
        saving_path = best_model_name_sweep.replace('logs/', '').replace('.pth', '.npy')
        test_metrics = evaluate_classifier(model, test_out_loader, param_pooling, device, save_path_preds=saving_path)
        print(test_metrics)

        print('{:1d}-Final: {:.7f}, Auc: {:.4f}, Acc: {:.4f}, Sens: {:.4f}, Speci: {:.4f}'
              ''.format(outer_split_num, test_metrics['loss'], test_metrics['auc'], test_metrics['acc'],
                        test_metrics['sensitivity'], test_metrics['specificity']))

    return test_metrics
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default="cuda:0")
    #parser.add_argument("--fold_num", type=int)
    parser.add_argument("--num_epochs", default=150, type=int)
    parser.add_argument("--target_var", default='gender')
    parser.add_argument("--num_nodes", default=50, type=int)
    parser.add_argument("--batch_size", default=500, type=int)
    parser.add_argument("--conn_type", default="fmri")
    parser.add_argument("--analysis_type", default='spatiotemporal')
    parser.add_argument("--time_length", default=1200, type=int)
    parser.add_argument("--remove_disconnected_nodes", default=False)
    parser.add_argument("--normalisation", default='subject_norm')
    parser.add_argument("--sweep_type", default='no_gnn')

    arguments = parser.parse_args()
    N_OUT_SPLITS = 5

    wandb.init(config={'test_results': True, 'sweep_type': arguments.sweep_type},
               name=f'final_{arguments.sweep_type}',
               project="spatio-temporal-brain")
    #config = wandb.config

    metrics=['auc', 'acc', 'f1', 'sensitivity', 'specificity']
    test_results = defaultdict(list)

    wandb_table = wandb.Table(columns=['Fold'] + metrics)
    for fold_num in range(1, N_OUT_SPLITS + 1):
        t_metrics = report_on_fold(fold_num, arguments)
        fold_ordered_metrics = [fold_num]
        for m in metrics:
            test_results[m].append(t_metrics[m])
            fold_ordered_metrics.append(round(t_metrics[m], 3))
            wandb.run.summary[f'{fold_num}_{m}'] = t_metrics[m]
        wandb_table.add_data(*fold_ordered_metrics)

    fold_ordered_mean = ['Mean']
    fold_ordered_std = ['Std']
    for m in metrics:
        final_mean = np.mean(test_results[m])
        final_std = np.std(test_results[m])

        fold_ordered_mean.append(round(final_mean, 2))
        fold_ordered_std.append(round(final_std, 2))
        print(m, ':', round(final_mean, 4), round(final_std, 4))

        wandb.run.summary[f'final_mean_{m}'] = final_mean
        wandb.run.summary[f'final_std_{m}'] = final_std

    wandb_table.add_data(*fold_ordered_mean)
    wandb_table.add_data(*fold_ordered_std)
    wandb.log({'summary_table': wandb_table})