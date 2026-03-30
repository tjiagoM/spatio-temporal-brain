import numpy as np
import torch
import wandb
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch
from tslearn.clustering import TimeSeriesKMeans

from datasets import BrainDataset
from main_loop import generate_dataset, create_fold_generator
from utils import change_w_config_, \
    get_freer_gpu

DEVICE_RUN = f'cpu'

runs_for_w_config = {0: {'run_id': 'wb93e183'},
                     1: {'run_id': '15jmbs0x'},
                     2: {'run_id': 'tr3362d3'},
                     3: {'run_id': '6gbh7jt1'},
                     4: {'run_id': '96wkdl8i'}}

if __name__ == '__main__':
    clusters = {
        0: torch.zeros((68, 68)).to(DEVICE_RUN),
        1: torch.zeros((68, 68)).to(DEVICE_RUN)
    }

    for fold_num in range(5):
        run_id = runs_for_w_config[fold_num]['run_id']
        api = wandb.Api()
        best_run = api.run(f'/tjiagom/st_extra/runs/{run_id}')
        w_config = best_run.config

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

            # X_train_out = dataset[torch.tensor(train_index)]
            X_test_out = dataset[torch.tensor(test_index)]

            break

        # Calculating on test set
        test_out_loader = DataLoader(X_test_out, batch_size=400, shuffle=False)

        num_clusters = 4
        for data in test_out_loader:
            print('.', end='')
            X, batch_mask = to_dense_batch(data.x, data.batch)

            with torch.no_grad():
                for batch_i in range(X.shape[0]):
                    sex_info = data.y[batch_i].item()
                    km = TimeSeriesKMeans(n_clusters=num_clusters)
                    labels = km.fit_predict(X[batch_i, :].cpu())
                    # Going over all the clusters found
                    for cluster_i in range(num_clusters):
                        cluster_ids = np.nonzero(labels == cluster_i)[0]
                        # Saving the matches for this cluster
                        for elem_i in range(cluster_ids.shape[0]):
                            for elem_j in range(elem_i + 1, cluster_ids.shape[0]):
                                clusters[sex_info][cluster_ids[elem_i], cluster_ids[elem_j]] += 1
        print()

    np.save(f'results/kmeans_clust_male.npy', clusters[1].cpu().numpy())
    np.save(f'results/kmeans_clust_female.npy', clusters[0].cpu().numpy())
