import numpy as np
import torch
import torch_geometric.utils as pyg_utils
import wandb
from torch_geometric.data import DataLoader

from datasets import BrainDataset
from main_loop import generate_dataset, create_fold_generator, generate_st_model
from model import SpatioTemporalModel
from utils import Normalisation, ConnType, DatasetType, \
    AnalysisType, EncodingStrategy, ConvStrategy, PoolingStrategy, SweepType, create_name_for_model, change_w_config_

DEVICE_RUN = 'cuda'

best_runs = {
    '100_n_diffpool': {0: {'run_id': 'khnljhrj'},
                       1: {'run_id': 'k9y54v5w', 'weight_d': 0.0012895162344404025},
                       2: {'run_id': '8ulilkox'},
                       3: {'run_id': 'm1lyyxez'},
                       4: {'run_id': 'lz7r38t4'}},
    '100_n_e_diffpool': {0: {'run_id': '1oysy05q'},
                         1: {'run_id': 'nxqb9kvj'},
                         2: {'run_id': 'skripjyc', 'weight_d': 1.5483273684368499e-06},
                         3: {'run_id': '6b3si6pc'},
                         4: {'run_id': 's1nhqmnj'}}
}

for model_type, runs_all in best_runs.items():
    print('----', model_type)
    clusters = {
        0: torch.zeros((68, 68)).to(DEVICE_RUN),
        1: torch.zeros((68, 68)).to(DEVICE_RUN)
    }
    total_amount = {
        0: 0,
        1: 0
    }

    for fold_num, run_info in runs_all.items():
        run_id = run_info['run_id']
        api = wandb.Api()
        best_run = api.run(f'/st-team/spatio-temporal-brain/runs/{run_id}')
        w_config = best_run.config

        change_w_config_(w_config)

        w_config['device_run'] = DEVICE_RUN
        if 'lr' in run_info.keys():
            w_config['param_lr'] = float(run_info['lr'])
        if 'dropout' in run_info.keys():
            w_config['param_dropout'] = float(run_info['dropout'])
        if 'weight_d' in run_info.keys():
            w_config['param_weight_decay'] = float(run_info['weight_d'])

        dataset: BrainDataset = generate_dataset(w_config)

        N_OUT_SPLITS = 5
        N_INNER_SPLITS = 5

        skf_outer_generator = create_fold_generator(dataset, w_config, N_OUT_SPLITS)

        # Getting train / test folds
        outer_split_num: int = 0
        for train_index, test_index in skf_outer_generator:
            outer_split_num += 1
            # Only run for the specific fold defined in the script arguments.
            if outer_split_num != w_config['fold_num']:
                continue

            X_test_out = dataset[torch.tensor(test_index)]

            break

        inner_fold_for_val: int = 1
        model: SpatioTemporalModel = generate_st_model(w_config, for_test=True)
        if 'model_v' in run_info.keys():
            model.VERSION = run_info['model_v']
        model_saving_path: str = create_name_for_model(target_var=w_config['target_var'],
                                                       model=model,
                                                       outer_split_num=w_config['fold_num'],
                                                       inner_split_num=inner_fold_for_val,
                                                       n_epochs=w_config['num_epochs'],
                                                       threshold=w_config['threshold'],
                                                       batch_size=w_config['batch_size'],
                                                       num_nodes=w_config['num_nodes'],
                                                       conn_type=w_config['param_conn_type'],
                                                       normalisation=w_config['param_normalisation'],
                                                       analysis_type=w_config['analysis_type'],
                                                       metric_evaluated='loss',
                                                       dataset_type=w_config['dataset_type'],
                                                       lr=w_config['param_lr'],
                                                       weight_decay=w_config['param_weight_decay'],
                                                       edge_weights=w_config['edge_weights'])
        if 'model_v' in run_info.keys():
            # We know the very specific "old" cases
            if w_config['param_pooling'] == PoolingStrategy.DIFFPOOL:
                model_saving_path = model_saving_path.replace('T_difW_F', 'GC_FGA_F')
            elif w_config['param_pooling'] == PoolingStrategy.MEAN:
                model_saving_path = model_saving_path.replace('T_no_W_F', 'GC_FGA_F')
        model.load_state_dict(torch.load(model_saving_path, map_location=w_config['device_run']))
        model.eval()

        # Calculating on test set
        # needs cast to int() because of higher precision when reading the csv
        test_out_loader = DataLoader(X_test_out, batch_size=400, shuffle=False)

        # TODO: change forward() function in model for this to be more straightforward instead of copying
        NEW_MAX_NODES = 17
        for data in test_out_loader:
            print('.', end='')
            with torch.no_grad():
                data = data.to(DEVICE_RUN)
                x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
                if model.multimodal_size > 0:
                    xn, x = x[:, :model.multimodal_size], x[:, model.multimodal_size:]
                    xn = model.multimodal_lin(xn)
                    xn = model.activation(xn)
                    xn = model.multimodal_batch(xn)
                    # xn = F.dropout(xn, p=self.dropout, training=self.training)

                # Processing temporal part
                if model.conv_strategy != ConvStrategy.NONE:
                    x = x.view(-1, 1, model.num_time_length)
                    x = model.temporal_conv(x)

                    # Concatenating for the final embedding per node
                    x = x.view(-1, model.size_before_lin_temporal)
                    x = model.lin_temporal(x)
                    x = model.activation(x)
                elif model.encoding_strategy == EncodingStrategy.STATS:
                    x = model.stats_lin(x)
                    x = model.activation(x)
                    x = model.stats_batch(x)

                if model.multimodal_size > 0:
                    x = torch.cat((xn, x), dim=1)

                if model.sweep_type in [SweepType.META_NODE, SweepType.META_EDGE_NODE]:
                    x, edge_attr, _ = model.meta_layer(x, edge_index, edge_attr)
                # print("Try 2")
                adj_tmp = pyg_utils.to_dense_adj(edge_index, data.batch, edge_attr=edge_attr)
                if edge_attr is not None:  # Because edge_attr only has 1 feature per edge
                    adj_tmp = adj_tmp[:, :, :, 0]
                x_tmp, batch_mask = pyg_utils.to_dense_batch(x, data.batch)
                # x, link_loss, ent_loss = model.diff_pool(x_tmp, adj_tmp, batch_mask)
                # print("Try 2")
                s = model.diff_pool.gnn1_pool(x_tmp, adj_tmp, batch_mask)
                s = s.unsqueeze(0) if s.dim() == 2 else s
                s = torch.softmax(s, dim=-1)

                # Going over each assigment matrix of the batch
                # TODO: stupidly slow, I needa optimise this, shouldn't never have done this this way
                for batch_i in range(s.shape[0]):
                    sex_info = data.y[batch_i].item()
                    total_amount[sex_info] += 1
                    for col in range(NEW_MAX_NODES):
                        matches = (s[batch_i][:, col] > 0.5).nonzero()
                        # Going over each match
                        for elem_i in range(matches.shape[0]):
                            for elem_j in range(elem_i + 1, matches.shape[0]):
                                clusters[sex_info][matches[elem_i].item(), matches[elem_j].item()] += 1
                                # print(matches[elem_i].item(), matches[elem_j].item())
                # break
                # np.save(f'diffpool_interp/s2_tmp{in_index}.npy', s.detach().cpu().numpy())
                # in_index += 1
        print()

    np.save(f'results/dp_interp_{model_type}_male.npy', clusters[1].cpu().numpy())
    np.save(f'results/dp_interp_{model_type}_female.npy', clusters[0].cpu().numpy())
    # a_female = clusters[0].numpy()
    # a_male = clusters[1].numpy()
