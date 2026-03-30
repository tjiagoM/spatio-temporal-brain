###
###
## Put this in root folder to be able to properly run it
###
import argparse

import numpy as np
from sklearn import svm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report


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

    X_train = np.load(f'data/svm_{dataset_type}_{analysis_type}_train_data_{fold_num}.npy')
    y_train = np.load(f'data/svm_{dataset_type}_{analysis_type}_train_label_{fold_num}.npy')
    X_val = np.load(f'data/svm_{dataset_type}_{analysis_type}_val_data_{fold_num}.npy')
    y_val = np.load(f'data/svm_{dataset_type}_{analysis_type}_val_label_{fold_num}.npy')
    X_test = np.load(f'data/svm_{dataset_type}_{analysis_type}_test_data_{fold_num}.npy')
    y_test = np.load(f'data/svm_{dataset_type}_{analysis_type}_test_label_{fold_num}.npy')

    # model = XGBClassifier(n_jobs=-1, tree_method='hist', n_estimators=5, verbosity=2,
    #                      random_state=111)
    # model.fit(X_train, y_train)
    best_val_metrics = {'acc': 0.0}
    corresponding_test_metrics = None
    for c_val in [1, 5, 10]:
        print(c_val, '...')
        model = svm.LinearSVC(random_state=111, C=c_val)
        # model = svm.SVC(cache_size=2000, random_state=111)#, max_iter=1000)
        model.fit(X_train, y_train)

        # print(model)

        # pickle.dump(model, open(f'tmp_SVM_model_{fold_num}.pkl', 'wb'))
        y_val_pred = model.predict(X_val)
        y_val_pred_bin = [round(value) for value in y_val_pred]
        val_metrics = return_classifier_metrics(y_val, y_val_pred_bin, y_val_pred)

        if val_metrics['acc'] > best_val_metrics['acc']:
            print('!')
            best_val_metrics = val_metrics

            y_pred = model.predict(X_test)
            y_pred_bin = [round(value) for value in y_pred]
            corresponding_test_metrics = return_classifier_metrics(y_test, y_pred_bin, y_pred)

    print(f'{dataset_type}_{analysis_type} / {fold_num}:', corresponding_test_metrics)


def parse_args():
    parser = argparse.ArgumentParser(description='Run baseline on flatten data for SVM')
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
