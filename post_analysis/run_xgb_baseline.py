import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier


def return_classifier_metrics(labels, pred_binary, pred_prob):
    roc_auc = roc_auc_score(labels, pred_prob)
    acc = accuracy_score(labels, pred_binary)
    f1 = f1_score(labels, pred_binary, zero_division=0)
    report = classification_report(labels, pred_binary, output_dict=True, zero_division=0)

    sens = report['1.0']['recall']
    spec = report['0.0']['recall']

    return {'auc': roc_auc,
            'acc': acc,
            'f1': f1,
            'sensitivity': sens,
            'specificity': spec
            }


def run_for_specific_fold(fold_num: int, dataset_type: str, analysis_type: str):
    print(f'RUNNING FOR {fold_num} on {dataset_type}...')

    # Take advantage of previously cached data from SVM baseline runs
    X_train = np.load(f'data/svm_{dataset_type}_{analysis_type}_train_data_{fold_num}.npy')
    y_train = np.load(f'data/svm_{dataset_type}_{analysis_type}_train_label_{fold_num}.npy')
    X_val = np.load(f'data/svm_{dataset_type}_{analysis_type}_val_data_{fold_num}.npy')
    y_val = np.load(f'data/svm_{dataset_type}_{analysis_type}_val_label_{fold_num}.npy')
    X_test = np.load(f'data/svm_{dataset_type}_{analysis_type}_test_data_{fold_num}.npy')
    y_test = np.load(f'data/svm_{dataset_type}_{analysis_type}_test_label_{fold_num}.npy')

    subsample_vals = np.random.uniform(0.4, 1, 25)
    max_depth_vals = np.random.choice(13, 25) + 3
    min_child_weights_vals = np.random.choice(10, 25) + 1
    colsubsample_bytree_vals = np.random.uniform(0.4, 1, 25)
    gamma_vals = np.random.choice(6, 25)

    best_val_metrics = {'acc': 0.0}
    corresponding_test_metrics = None
    i = 0
    for sub, max_d, min_c, cols, gamma in zip(subsample_vals, max_depth_vals, min_child_weights_vals,
                                              colsubsample_bytree_vals, gamma_vals):
        print(f'Trial {i}...')
        model = XGBClassifier(subsample=sub,
                              max_depth=max_d,
                              min_child_weight=min_c,
                              colsample_bytree=cols,
                              gamma=gamma,
                              n_jobs=-1,
                              random_state=1111)

        model.fit(X_train, y_train)

        y_val_pred_bin = model.predict(X_val)
        y_val_pred = model.predict_proba(X_val)[:, 1]
        val_metrics = return_classifier_metrics(y_val, y_val_pred_bin, y_val_pred)

        if val_metrics['acc'] > best_val_metrics['acc']:
            print('!')
            best_val_metrics = val_metrics

            y_pred_bin = model.predict(X_test)
            y_pred = model.predict_proba(X_test)[:, 1]
            corresponding_test_metrics = return_classifier_metrics(y_test, y_pred_bin, y_pred)

        i += 1

    best_file = open(f'results/xgb_{dataset_type}_{analysis_type}_test_{fold_num}.json', 'w')
    json.dump(corresponding_test_metrics, best_file)
    best_file.close()

    print(f'{dataset_type}_{analysis_type} / {fold_num}:', corresponding_test_metrics)


def parse_args():
    parser = argparse.ArgumentParser(description='Run baseline on flatten data for UK Biobank')
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
