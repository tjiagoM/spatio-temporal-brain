import torch
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import DataLoader, DenseDataLoader
import numpy as np
import torch_geometric.utils as pyg_utils
from datasets import BrainDataset
from model import SpatioTemporalModel
from utils import create_name_for_hcp_dataset, Normalisation, ConnType, ConvStrategy, PoolingStrategy, \
    StratifiedGroupKFold
import torch.nn.functional as F

device = 'cuda'

#N_EPOCHS = 1
TARGET_VAR = 'gender'
#ACTIVATION = 'relu'
THRESHOLD = 20
SPLIT_TO_TEST = 1
#ADD_GCN = False
#ADD_GAT = False
BATCH_SIZE = 500
REMOVE_NODES = False
NUM_NODES = 50
CONN_TYPE = ConnType('fmri')
#CONV_STRATEGY = ConvStrategy('entire')
#POOLING = PoolingStrategy('diff_pool')
#CHANNELS_CONV = 8
NORMALISATION = Normalisation('roi_norm')

model_path = 'logs/gender_1_0_auc_D_0.5__A_relu__P_diff_pool__CS_entire__CHC_8__FS_True__GCN_False__GAT_False_0.0001_0_30_5_roi_norm_500_False_50_fmri.pth'

def merge_y_and_others(ys, sessions, directions):
    tmp = torch.cat([ys.long().view(-1, 1),
                     sessions.view(-1, 1),
                     directions.view(-1, 1)], dim=1)
    return LabelEncoder().fit_transform([str(l) for l in tmp.numpy()])


name_dataset = create_name_for_hcp_dataset(num_nodes=NUM_NODES,
                                               target_var=TARGET_VAR,
                                               threshold=THRESHOLD,
                                               normalisation=NORMALISATION,
                                               connectivity_type=CONN_TYPE,
                                               disconnect_nodes=REMOVE_NODES)
print("Going for", name_dataset)
dataset = BrainDataset(root=name_dataset,
                       num_nodes=NUM_NODES,
                       target_var=TARGET_VAR,
                       threshold=THRESHOLD,
                       normalisation=NORMALISATION,
                       connectivity_type=CONN_TYPE,
                       disconnect_nodes=REMOVE_NODES)

N_OUT_SPLITS = 5
N_INNER_SPLITS = 5

if TARGET_VAR == 'gender':
    # Stratification will occur with regards to both the sex and session day
    skf = StratifiedGroupKFold(n_splits=N_OUT_SPLITS, random_state=1111)
    merged_labels = merge_y_and_others(dataset.data.y,
                                       dataset.data.session,
                                       dataset.data.direction)
    skf_generator = skf.split(np.zeros((len(dataset), 1)),
                              merged_labels,
                              groups=dataset.data.hcp_id.tolist())


outer_split_num = 0
for train_index, test_index in skf_generator:
    outer_split_num += 1
    # Only run for the specific fold defined in the script arguments.
    if outer_split_num != SPLIT_TO_TEST:
        continue
    X_train_out = dataset[torch.tensor(train_index)]
    X_test_out = dataset[torch.tensor(test_index)]
    print("Size is:", len(X_train_out), "/", len(X_test_out))
    print("Positive classes:", sum(X_train_out.data.y.numpy()), "/", sum(X_test_out.data.y.numpy()))
    train_out_loader = DataLoader(X_train_out, batch_size=BATCH_SIZE, shuffle=True)
    test_out_loader = DataLoader(X_test_out, batch_size=BATCH_SIZE, shuffle=True)
    model = torch.load(model_path, map_location=torch.device(device))
    # Getting the values to the model.
    model.eval()
    criterion = torch.nn.BCELoss()
    predictions = []
    labels = []
    test_error = 0
    in_index = 0
    for data in test_out_loader:
        with torch.no_grad():
            data = data.to(device)
            print("Try 1")
            x, edge_index = data.x, data.edge_index
            x = x.view(-1, 1, model.num_time_length)
            x = model.temporal_conv(x)
            x = x.view(-1, model.channels_conv * 8 * model.final_feature_size)
            x = model.lin_temporal(x)
            x = model.activation(x)
            x = F.dropout(x, p=model.dropout, training=model.training)
            print("Try 2")
            adj_tmp = pyg_utils.to_dense_adj(edge_index, data.batch)
            x_tmp, batch_mask = pyg_utils.to_dense_batch(x, data.batch)
            #x, link_loss, ent_loss = model.diff_pool(x_tmp, adj_tmp, batch_mask)
            print("Try 2")
            s = model.diff_pool.gnn1_pool(x_tmp, adj_tmp, batch_mask)
            s = s.unsqueeze(0) if s.dim() == 2 else s
            s = torch.softmax(s, dim=-1)
            np.save(f's2_tmp{in_index}.npy', s.detach().cpu().numpy())
            in_index += 1
