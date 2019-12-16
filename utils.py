import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import RobustScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def create_name_for_hcp_dataset(num_nodes, target_var, threshold, connectivity_type,
                                disconnect_nodes=False,
                                prefix_location='./pytorch_data/hcp_'):
    name_combination = '_'.join([target_var, connectivity_type, str(num_nodes), str(threshold), str(disconnect_nodes)])

    return prefix_location + name_combination


def create_name_for_model(target_var, model, params, outer_split_num, inner_split_num, n_epochs, threshold, batch_size,
                          remove_disconnect_nodes, num_nodes, conn_type,
                          metric_evaluated,
                          prefix_location='logs/'):
    return prefix_location + '_' + '_'.join([target_var,
                                             str(outer_split_num),
                                             str(inner_split_num),
                                             metric_evaluated,
                                             model.to_string_name(),
                                             str(params['lr']),
                                             str(params['weight_decay']),
                                             str(n_epochs),
                                             str(threshold),
                                             str(batch_size),
                                             str(remove_disconnect_nodes),
                                             str(num_nodes),
                                             str(conn_type)
                                             ]) + '.pth'
