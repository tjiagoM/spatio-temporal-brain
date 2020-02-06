import torch
from torch_geometric.data import DataLoader, DenseDataLoader
import numpy as np

from datasets import HCPDataset
from model import SpatioTemporalModel
from utils import create_name_for_hcp_dataset, Normalisation, ConnType, ConvStrategy, PoolingStrategy

device = 'cuda:1'

N_EPOCHS = 1
TARGET_VAR = 'gender'
ACTIVATION = 'relu'
THRESHOLD = 5
SPLIT_TO_TEST = 1
ADD_GCN = False
ADD_GAT = False
BATCH_SIZE = 150
REMOVE_NODES = False
NUM_NODES = 50
CONN_TYPE = ConnType('fmri')
CONV_STRATEGY = ConvStrategy('entire')
POOLING = PoolingStrategy('mean')
CHANNELS_CONV = 8
NORMALISATION = Normalisation('roi_norm')
TIME_LENGTH = 1200

torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

name_dataset = create_name_for_hcp_dataset(num_nodes=NUM_NODES,
                                           time_length=1200,
                                           target_var=TARGET_VAR,
                                           threshold=THRESHOLD,
                                           normalisation=NORMALISATION,
                                           connectivity_type=CONN_TYPE,
                                           disconnect_nodes=REMOVE_NODES)
print("Going for", name_dataset)
dataset = HCPDataset(root=name_dataset,
                     time_length=1200,
                     num_nodes=NUM_NODES,
                     target_var=TARGET_VAR,
                     threshold=THRESHOLD,
                     normalisation=NORMALISATION,
                     connectivity_type=CONN_TYPE,
                     disconnect_nodes=REMOVE_NODES)

model = SpatioTemporalModel(num_time_length=TIME_LENGTH,
                            dropout_perc=0.3,
                            pooling=POOLING,
                            channels_conv=CHANNELS_CONV,
                            activation=ACTIVATION,
                            conv_strategy=CONV_STRATEGY,
                            add_gat=ADD_GAT,
                            add_gcn=ADD_GCN,
                            final_sigmoid=True,
                            num_nodes=NUM_NODES
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

