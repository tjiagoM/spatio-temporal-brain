from torch_geometric.data import DataLoader

from datasets import HCPDataset
from model import SpatioTemporalModel
from utils import create_name_for_hcp_dataset

N_EPOCHS = 1
TARGET_VAR = 'gender'
ACTIVATION = 'relu'
THRESHOLD = 20
SPLIT_TO_TEST = 1
ADD_GCN = True
ADD_GAT = False
BATCH_SIZE = 150
REMOVE_NODES = False
NUM_NODES = 272
CONN_TYPE = 'struct'
CONV_STRATEGY = 'entire'
POOLING = 'mean'
CHANNELS_CONV = 1

name_dataset = create_name_for_hcp_dataset(num_nodes=NUM_NODES,
                                               target_var=TARGET_VAR,
                                               threshold=THRESHOLD,
                                               connectivity_type=CONN_TYPE,
                                               disconnect_nodes=REMOVE_NODES)
print("Going for", name_dataset)
dataset = HCPDataset(root=name_dataset,
                     num_nodes=NUM_NODES,
                     target_var=TARGET_VAR,
                     threshold=THRESHOLD,
                     connectivity_type=CONN_TYPE,
                     disconnect_nodes=REMOVE_NODES)

model = SpatioTemporalModel(num_time_length=2400,
                             dropout_perc=0.3,
                             pooling=POOLING,
                             channels_conv=CHANNELS_CONV,
                             activation=ACTIVATION,
                             conv_strategy=CONV_STRATEGY,
                             add_gat=ADD_GAT,
                             add_gcn=ADD_GCN,
                             final_sigmoid=True
                             ).to('cpu')

train_out_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model.train()
loss_all = 0
    criterion = torch.nn.BCELoss()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output_batch = model(data)
        loss = criterion(output_batch, data.y.unsqueeze(1))
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

    # len(train_loader) gives the number of batches
    # len(train_loader.dataset) gives the number of graphs

    # Returning a weighted average according to number of graphs
    return loss_all / len(train_loader.dataset)