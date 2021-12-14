import numpy as np
import torch
import torch_geometric.utils as pyg_utils
import wandb
from torch_geometric.data import DataLoader

from datasets import BrainDataset
from main_loop import generate_dataset, create_fold_generator, generate_st_model
from model import SpatioTemporalModel
from utils import Normalisation, ConnType, DatasetType, \
    AnalysisType, EncodingStrategy, ConvStrategy, PoolingStrategy, SweepType, create_name_for_model, change_w_config_, \
    get_freer_gpu, calculate_indegree_histogram

DEVICE_RUN = f'cuda:{get_freer_gpu()}'

best_runs = {
    '100_n_e_diffpool': {0: {'run_id': 'wb93e183'},
                                      1: {'run_id': '15jmbs0x'},
                                      2: {'run_id': 'tr3362d3'},
                                      3: {'run_id': '6gbh7jt1'},
                                      4: {'run_id': '96wkdl8i'}},
    '100_n_diffpool': {0: {'run_id': 'v0nljvcf'},
                                  1: {'run_id': 'dncxffke'},
                                  2: {'run_id': 'dvc6767t'},
                                  3: {'run_id': 'vliojzuh'},
                                  4: {'run_id': 's1cjijtu'}}
}

if __name__ == '__main__':
    import os
    os.chdir('..')
    print(os.getcwd())

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
            best_run = api.run(f'/tjiagom/st_extra/runs/{run_id}')
            w_config = best_run.config

            model_file_name = None
            for file in best_run.files():
                if file.name.endswith('.pt'):
                    model_file_name = file.name
                    file.download(replace=True)
                    break

            change_w_config_(w_config)
            w_config['device_run'] = DEVICE_RUN

            dataset: BrainDataset = generate_dataset(w_config)

            N_OUT_SPLITS: int = 5
            N_INNER_SPLITS: int = 5

            skf_outer_generator = create_fold_generator(dataset, w_config, N_OUT_SPLITS)

            # Getting train / test folds
            outer_split_num: int = 0
            for train_index, test_index in skf_outer_generator:
                outer_split_num += 1
                # Only run for the specific fold defined in the script arguments.
                if outer_split_num != w_config['fold_num']:
                    continue

                X_train_out = dataset[torch.tensor(train_index)]
                X_test_out = dataset[torch.tensor(test_index)]

                break

            skf_inner_generator = create_fold_generator(X_train_out, w_config, N_INNER_SPLITS)
            inner_loop_run: int = 0
            for inner_train_index, inner_val_index in skf_inner_generator:
                inner_loop_run += 1

                X_train_in = X_train_out[torch.tensor(inner_train_index)]
                #X_val_in = X_train_out[torch.tensor(inner_val_index)]

                w_config['dataset_indegree'] = calculate_indegree_histogram(X_train_in)

                # model: SpatioTemporalModel = generate_st_model(run_cfg)
                model = SpatioTemporalModel(run_cfg=w_config,
                                            encoding_model=None
                                            ).to(w_config['device_run'])

                break
            model.load_state_dict(torch.load(model_file_name, map_location=w_config['device_run']))
            model.eval()

            # Calculating on test set
            # needs cast to int() because of higher precision when reading the csv
            test_out_loader = DataLoader(X_test_out, batch_size=400, shuffle=False)

            # TODO: change forward() function in model for this to be more straightforward instead of copying
            NEW_MAX_NODES = 48 # ceil(0.7 * 68)
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
                    x = x.view(-1, 1, model.num_time_length)
                    x = model.temporal_conv(x)

                    # Concatenating for the final embedding per node
                    x = x.view(-1, model.size_before_lin_temporal)
                    x = model.lin_temporal(x)
                    x = model.activation(x)

                    if model.multimodal_size > 0:
                        x = torch.cat((xn, x), dim=1)

                    if model.sweep_type == SweepType.META_NODE:
                        x = model.meta_layer(x, edge_index, edge_attr)
                    elif model.sweep_type == SweepType.META_EDGE_NODE:
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
