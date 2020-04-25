import torch
from torch_geometric.data import DataLoader, DenseDataLoader
import numpy as np

from datasets import BrainDataset, HCPDataset
from model import SpatioTemporalModel
from utils import create_name_for_brain_dataset, Normalisation, ConnType, ConvStrategy, PoolingStrategy, \
    EncodingStrategy, \
    create_best_encoder_name, AnalysisType, DatasetType

device = 'cuda:1'

N_EPOCHS = 1
TARGET_VAR = 'gender'
ACTIVATION = 'relu'
THRESHOLD = 40
SPLIT_TO_TEST = 1
ADD_GCN = False
ADD_GAT = False
BATCH_SIZE = 200
REMOVE_NODES = False
NUM_NODES = 68
CONN_TYPE = ConnType('fmri')
CONV_STRATEGY = ConvStrategy('tcn_entire')
POOLING = PoolingStrategy('mean')
CHANNELS_CONV = 8
NORMALISATION = Normalisation('subject_norm')
TIME_LENGTH = 1200
ENCODING_STRATEGY = EncodingStrategy('none')
DATASET_TYPE = DatasetType.HCP
ANALYSIS_TYPE = AnalysisType.ST_MULTIMODAL
MULTIMODAL_SIZE = 10

torch.manual_seed(1)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

name_dataset = create_name_for_brain_dataset(num_nodes=NUM_NODES,
                                                 time_length=TIME_LENGTH,
                                                 target_var=TARGET_VAR,
                                                 threshold=THRESHOLD,
                                                 normalisation=Normalisation.SUBJECT,
                                                 connectivity_type=ConnType.STRUCT,
                                                 analysis_type=ANALYSIS_TYPE,
                                             encoding_strategy=ENCODING_STRATEGY,
                                                 dataset_type=DATASET_TYPE)

class_dataset = HCPDataset

dataset = class_dataset(root=name_dataset,
                            target_var=TARGET_VAR,
                            num_nodes=NUM_NODES,
                            threshold=THRESHOLD,
                            connectivity_type=ConnType.STRUCT,
                            normalisation=Normalisation.SUBJECT,
                            analysis_type=ANALYSIS_TYPE,
                        encoding_strategy=ENCODING_STRATEGY,
                            time_length=TIME_LENGTH)
#if ENCODING_STRATEGY != EncodingStrategy.NONE:
#    from encoders import AE # Necessary to load
#    encoding_model = torch.load(create_best_encoder_name(ts_length=TIME_LENGTH,
#                                                         outer_split_num=SPLIT_TO_TEST,
#                                                         encoder_name=ENCODING_STRATEGY.value))
#else:
#    encoding_model = None
encoding_model = None
model = SpatioTemporalModel(num_time_length=TIME_LENGTH,
                                dropout_perc=0.3,
                                pooling=POOLING,
                                channels_conv=CHANNELS_CONV,
                                activation=ACTIVATION,
                                conv_strategy=CONV_STRATEGY,
                                add_gat=ADD_GAT,
                                gat_heads=0,
                                add_gcn=ADD_GCN,
                                final_sigmoid=True,
                                num_nodes=NUM_NODES,
                                num_gnn_layers=0,
                                multimodal_size=MULTIMODAL_SIZE,
                            encoding_strategy=ENCODING_STRATEGY,
                                encoding_model=None
                                ).to(device)
pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_trainable_params)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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
#pip install nolds
#
#git clone https://github.com/raphaelvallat/entropy.git entropy/
#cd entropy/
#pip install -r requirements.txt
#python setup.py develop
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



#unique_people = []
#unique_y = []
#for person_id, outcome in zip(dataset.data.hcp_id.tolist(), dataset.data.y.tolist()):
#    if person_id not in unique_people:
#        unique_people.append(person_id)
#        unique_y.append(outcome)
from sklearn.preprocessing import LabelEncoder

def merge_y_and_session(ys, sessions, directions):
    tmp = torch.cat([ys.long().view(-1, 1),
                     sessions.view(-1, 1),
                     directions.view(-1,1)], dim=1)
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

