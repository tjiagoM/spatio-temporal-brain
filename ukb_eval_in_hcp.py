import pickle
import wandb
import numpy as np
from xgboost import XGBClassifier

from datasets import FlattenCorrsDataset
from main_loop import generate_xgb_model, return_metrics, send_global_results
from utils import DatasetType, AnalysisType, ConnType, create_name_for_flattencorrs_dataset, create_name_for_xgbmodel

RUN_NAME = 'ukb_eval_on_hcp_flatten'

api = wandb.Api()
best_run = api.run("/st-team/spatio-temporal-brain/runs/jkko3173")
w_config = best_run.config

w_config['analysis_type'] = AnalysisType(w_config['analysis_type'])
w_config['dataset_type'] = DatasetType(w_config['dataset_type'])
w_config['param_conn_type'] = ConnType(w_config['conn_type'])

# Getting best model
inner_fold_for_val: int = 1
model: XGBClassifier = generate_xgb_model(w_config)
model_saving_path = create_name_for_xgbmodel(model=model,
                                             outer_split_num=w_config['fold_num'],
                                             inner_split_num=inner_fold_for_val,
                                             run_cfg=best_run.config
                                             )
model = pickle.load(open(model_saving_path, "rb"))

# Getting HCP Data
hcp_dict = {
'dataset_type': DatasetType('hcp'),
'analysis_type' : AnalysisType('flatten_corrs'),
'param_conn_type' : ConnType('fmri'),
'num_nodes': 68,
'time_length': 1200
}
name_dataset = create_name_for_flattencorrs_dataset(hcp_dict)
dataset = FlattenCorrsDataset(root=name_dataset,
                              num_nodes=68,
                              connectivity_type=ConnType('fmri'),
                              analysis_type=AnalysisType('flatten_corrs'),
                              dataset_type=DatasetType('hcp'),
                              time_length=1200)
hcp_arr = np.array([data.x.numpy() for data in dataset])
hcp_y_test = [int(data.sex.item()) for data in dataset]

test_metrics = return_metrics(hcp_y_test,
                              pred_prob=model.predict_proba(hcp_arr)[:, 1],
                              pred_binary=model.predict(hcp_arr),
                              flatten_approach=True)
print(test_metrics)

wandb.init(entity='st-team', name=RUN_NAME)

send_global_results(test_metrics)