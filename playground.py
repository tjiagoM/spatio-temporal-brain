import numpy as np
import torch
from torch_geometric.data import DataLoader

from main_loop import generate_dataset, create_fold_generator, generate_st_model
from model import SpatioTemporalModel
from utils import Normalisation, ConnType, ConvStrategy, PoolingStrategy, EncodingStrategy, \
    AnalysisType, DatasetType, SweepType

run_cfg = {'analysis_type': AnalysisType.ST_UNIMODAL,
           'dataset_type': DatasetType.UKB,
           'num_nodes': 68,
           'param_conn_type': ConnType('fmri'),
           'split_to_test': 1,
           'sweep_type': SweepType('edge_node_meta'),
           'target_var': 'gender',
           'time_length': 490,
           'param_gat_heads': 0,

           'batch_size': 400,
           'device_run': 'cuda:1',
           # 'early_stop_steps': config.early_stop_steps,
           'edge_weights': True,
           'model_with_sigmoid': True,
           'num_epochs': 1,
           'param_activation': 'relu',
           'param_channels_conv': 8,
           'param_conv_strategy': ConvStrategy('tcn_entire'),
           'param_dropout': 0.3,
           'param_encoding_strategy': EncodingStrategy('none'),
           # 'param_lr': config.lr,
           'param_normalisation': Normalisation('subject_norm'),
           'param_num_gnn_layers': 1,
           'param_pooling': PoolingStrategy('mean'),
           'param_threshold': 5,
           # 'param_weight_decay': config.weight_decay,
           'temporal_embed_size': 128,

           'multimodal_size': 0
           }
run_cfg['ts_spit_num'] = int(4800 / run_cfg['time_length'])

torch.manual_seed(1)

N_OUT_SPLITS: int = 5
N_INNER_SPLITS: int = 5

dataset = generate_dataset(run_cfg)

skf_outer_generator = create_fold_generator(dataset, run_cfg['dataset_type'], run_cfg['analysis_type'], N_OUT_SPLITS)

outer_split_num: int = 0
for train_index, test_index in skf_outer_generator:
    outer_split_num += 1
    # Only run for the specific fold defined in the script arguments.
    if outer_split_num != run_cfg['split_to_test']:
        continue

    X_train_out = dataset[torch.tensor(train_index)]
    X_test_out = dataset[torch.tensor(test_index)]

    break

model: SpatioTemporalModel = generate_st_model(run_cfg, for_test=True)

train_loader = DataLoader(X_train_out, batch_size=run_cfg['batch_size'], shuffle=True)

model.train()
loss_all = 0
criterion = torch.nn.BCELoss()

for data in train_loader:
    data = data.to(device)
    if POOLING == PoolingStrategy.DIFFPOOL:
        output_batch, link_loss, ent_loss = model(data)
        loss = criterion(output_batch, data.y.unsqueeze(1)) + link_loss + ent_loss
    else:
        output_batch = model(data)
        loss = criterion(output_batch, data.y.unsqueeze(1))

    loss.backward()
    break

    loss_all += loss.item() * data.num_graphs
    optimizer.step()

    # len(train_loader) gives the number of batches
    # len(train_loader.dataset) gives the number of graphs

exit()

#############################################################
#
# pip install nolds
#
# git clone https://github.com/raphaelvallat/entropy.git entropy/
# cd entropy/
# pip install -r requirements.txt
# python setup.py develop
# (I had to install numba through conda in order for entropy to work)

from scipy.stats import skew, kurtosis
from entropy import app_entropy, perm_entropy, sample_entropy, spectral_entropy, svd_entropy, \
    detrended_fluctuation, higuchi_fd, katz_fd, petrosian_fd
import nolds

timeseries = dataset[0].x[:, 10:].numpy()

# Size: (68,)
means = timeseries.mean(axis=1)
variances = timeseries.std(axis=1)
mins = timeseries.min(axis=1)
maxs = timeseries.max(axis=1)
skewnesses = skew(timeseries, axis=1)
kurtos = kurtosis(timeseries, axis=1, bias=False)
# Approximate entropy
entro_app = np.apply_along_axis(app_entropy, 1, timeseries)
# Permutation Entropy
entro_perm = np.apply_along_axis(perm_entropy, 1, timeseries, normalize=True)
# Sample Entropy
entro_sample = np.apply_along_axis(sample_entropy, 1, timeseries)
# Spectral Entropy with Fourier Transform
entro_spectr = np.apply_along_axis(spectral_entropy, 1, timeseries, sf=1, normalize=True)
# Singular Value Decomposition entropy
entro_svd = np.apply_along_axis(svd_entropy, 1, timeseries, normalize=True)
# Detrended fluctuation analysis (DFA)
fractal_dfa = np.apply_along_axis(detrended_fluctuation, 1, timeseries)
# Higuchi Fractal Dimension
fractal_higuchi = np.apply_along_axis(higuchi_fd, 1, timeseries)
# Katz Fractal Dimension.
fractal_katz = np.apply_along_axis(katz_fd, 1, timeseries)
# Petrosian fractal dimension
fractal_petro = np.apply_along_axis(petrosian_fd, 1, timeseries)
# Hurst Exponent
hursts = np.apply_along_axis(nolds.hurst_rs, 1, timeseries)

merged_stats = (means, variances, mins, maxs, skewnesses, kurtos, entro_app, entro_perm, entro_sample, entro_spectr,
                entro_svd, fractal_dfa, fractal_higuchi, fractal_katz, fractal_petro, hursts)
merged_stats = np.vstack(merged_stats).T
assert merged_stats.shape == (68, 16)

#########################


# unique_people = []
# unique_y = []
# for person_id, outcome in zip(dataset.data.hcp_id.tolist(), dataset.data.y.tolist()):
#    if person_id not in unique_people:
#        unique_people.append(person_id)
#        unique_y.append(outcome)
from sklearn.preprocessing import LabelEncoder


def merge_y_and_session(ys, sessions, directions):
    tmp = torch.cat([ys.long().view(-1, 1),
                     sessions.view(-1, 1),
                     directions.view(-1, 1)], dim=1)
    return LabelEncoder().fit_transform([str(l) for l in tmp.numpy()])


y_final = merge_y_and_session(dataset.data.y,
                              dataset.data.session,
                              dataset.data.direction)

from sklearn.model_selection import GroupKFold

skf = GroupKFold(n_splits=10)
skf_generator = skf.split(np.zeros((len(dataset), 1)),
                          y=y_final,
                          groups=dataset.data.hcp_id.tolist())

split_num = 0
for train_index, test_index in skf_generator:
    train_data = dataset[torch.tensor(train_index)]
    test_data = dataset[torch.tensor(test_index)]

    len_train_y1 = len(train_data.data.y[train_data.data.y == 1])
    len_train_y0 = len(train_data.data.y[train_data.data.y == 0])

    len_test_y1 = len(test_data.data.y[test_data.data.y == 1])
    len_test_y0 = len(test_data.data.y[test_data.data.y == 0])

    uniq_train_people = train_data.data.hcp_id.tolist()
    uniq_test_people = test_data.data.hcp_id.tolist()
    num_intersection = len(set(uniq_test_people).intersection(set(uniq_train_people)))

    len_train_lr = len(train_data.data.direction[train_data.data.direction == 1])
    len_train_rl = len(train_data.data.direction[train_data.data.direction == 0])

    len_test_lr = len(test_data.data.direction[test_data.data.direction == 1])
    len_test_rl = len(test_data.data.direction[test_data.data.direction == 0])

    print("For\t", split_num, "\t", len(train_data), "\t", len(test_data),
          "\t", len_train_y1, "/", len_train_y0,
          "\t", len_test_y1, "/", len_test_y0,
          "\t", num_intersection,
          "\t", len_train_lr, "/", len_train_lr,
          "\t", len_test_rl, "/", len_test_rl)

    split_num += 1

print()
############################################################################
############################################################################
############################################################################
import numpy as np
from utils import StratifiedGroupKFold

split_num = 0
skf = StratifiedGroupKFold(n_splits=10, random_state=1112)
skf_generator = skf.split(np.zeros((len(dataset), 1)),
                          y_final,
                          groups=dataset.data.hcp_id.tolist())
for train_index, test_index in skf_generator:
    train_data = dataset[torch.tensor(train_index)]
    test_data = dataset[torch.tensor(test_index)]

    len_train_y1 = len(train_data.data.y[train_data.data.y == 1])
    len_train_y0 = len(train_data.data.y[train_data.data.y == 0])

    len_test_y1 = len(test_data.data.y[test_data.data.y == 1])
    len_test_y0 = len(test_data.data.y[test_data.data.y == 0])

    uniq_train_people = train_data.data.hcp_id.tolist()
    uniq_test_people = test_data.data.hcp_id.tolist()
    num_intersection = len(set(uniq_test_people).intersection(set(uniq_train_people)))

    len_train_lr = len(train_data.data.direction[train_data.data.direction == 1])
    len_train_rl = len(train_data.data.direction[train_data.data.direction == 0])

    len_test_lr = len(test_data.data.direction[test_data.data.direction == 1])
    len_test_rl = len(test_data.data.direction[test_data.data.direction == 0])

    print("For\t", split_num, "\t", len(train_data), "\t", len(test_data),
          "\t", len_train_y1, "/", len_train_y0,
          "\t", len_test_y1, "/", len_test_y0,
          "\t", num_intersection,
          "\t", len_train_lr, "/", len_train_lr,
          "\t", len_test_rl, "/", len_test_rl)

    split_num += 1
