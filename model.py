from sys import exit
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch_geometric.nn import global_mean_pool, GCNConv
import torch_geometric.utils as pyg_utils
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool

from utils import ConvStrategy, PoolingStrategy

from torch.nn.utils import weight_norm
from tcn import TemporalConvNet


# function to extract grad
def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook

class GNN(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 normalize=False,
                 add_loop=False,
                 lin=True):
        super(GNN, self).__init__()

        self.add_loop = add_loop

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask, self.add_loop)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask, self.add_loop)))
        x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask, self.add_loop)))

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x


class DiffPoolLayer(torch.nn.Module):
    def __init__(self, max_num_nodes, num_init_feats):
        super(DiffPoolLayer, self).__init__()
        self.init_feats = num_init_feats
        self.max_nodes = max_num_nodes
        self.INTERN_EMBED_SIZE = ceil(self.init_feats / 3)

        num_nodes = ceil(0.25 * self.max_nodes)
        self.gnn1_pool = GNN(self.init_feats, self.INTERN_EMBED_SIZE, num_nodes, add_loop=True)
        self.gnn1_embed = GNN(self.init_feats, self.INTERN_EMBED_SIZE, self.INTERN_EMBED_SIZE, add_loop=True, lin=True)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(self.INTERN_EMBED_SIZE, self.INTERN_EMBED_SIZE, num_nodes)
        self.gnn2_embed = GNN(self.INTERN_EMBED_SIZE, self.INTERN_EMBED_SIZE, self.INTERN_EMBED_SIZE, lin=True)

        self.gnn3_embed = GNN(self.INTERN_EMBED_SIZE, self.INTERN_EMBED_SIZE, self.INTERN_EMBED_SIZE, lin=True)

        #self.lin1 = torch.nn.Linear(3 * 64, 64)
        #self.lin2 = torch.nn.Linear(self.INTERN_EMBED_SIZE, 1)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)

        x = self.gnn1_embed(x, adj, mask)


        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)


        x = x.mean(dim=1)

        #x = F.relu(self.lin1(x))
        # dropout here?
        #print("4", x.shape)
        #x = self.lin2(x)
        #print("5", x.shape)
        return x, l1 + l2, e1 + e2


class SpatioTemporalModel(nn.Module):
    def __init__(self, num_time_length, dropout_perc, pooling, channels_conv, activation, conv_strategy,
                 add_gat=False, add_gcn=False, final_sigmoid=True):
        super(SpatioTemporalModel, self).__init__()

        if pooling not in [PoolingStrategy.DIFFPOOL, PoolingStrategy.DIFFPOOL]:
            print("THIS IS NOT PREPARED FOR OTHER POOLING THAN MEAN/DIFFPOOL")
            exit(-1)
        if conv_strategy not in [ConvStrategy.CNN_2, ConvStrategy.TCN_2, ConvStrategy.ENTIRE]:
            print("THIS IS NOT PREPARED FOR OTHER CONV STRATEGY THAN ENTIRE/2_conv/2_tcn")
            exit(-1)
        if activation not in ['relu', 'tanh', 'elu']:
            print("THIS IS NOT PREPARED FOR OTHER ACTIVATION THAN relu/tanh/elu")
            exit(-1)
        if add_gat and add_gcn:
            print("You cannot have both GCN and GAT")
            exit(-1)

        self.TEMPORAL_EMBED_SIZE = 256
        self.dropout = dropout_perc
        self.pooling = pooling
        dict_activations = {'relu': nn.ReLU(),
                            'elu': nn.ELU(),
                            'tanh': nn.Tanh()}
        self.activation = dict_activations[activation]
        self.activation_str = activation

        self.conv_strategy = conv_strategy

        self.channels_conv = channels_conv
        self.final_channels = 1 if channels_conv == 1 else channels_conv * 2
        self.final_sigmoid = final_sigmoid
        self.add_gcn = add_gcn
        self.add_gat = add_gat  # TODO

        self.num_time_length = num_time_length
        self.final_feature_size = round(self.num_time_length / 2 / 8)
        #self.final_feature_size = 7

        self.gcn_conv1 = GCNConv(self.final_feature_size * self.final_channels,
                                 self.final_feature_size * self.final_channels)

        # CNNs for temporal domain
        self.conv1d_1 = nn.Conv1d(1, self.channels_conv, 7, padding=3, stride=2)
        self.conv1d_2 = nn.Conv1d(self.channels_conv, self.channels_conv * 2, 7, padding=3, stride=2)
        self.conv1d_3 = nn.Conv1d(self.channels_conv * 2, self.channels_conv * 4, 7, padding=3, stride=2)
        self.conv1d_4 = nn.Conv1d(self.channels_conv * 4, self.channels_conv * 8, 7, padding=3, stride=2)
        self.batch1 = BatchNorm1d(self.channels_conv)
        self.batch2 = BatchNorm1d(self.channels_conv * 2)
        self.batch3 = BatchNorm1d(self.channels_conv * 4)
        self.batch4 = BatchNorm1d(self.channels_conv * 8)

        #if self.conv_strategy == ConvStrategy.ENTIRE:
        #    self.conv1d_4 = torch.nn.Conv1d(self.channels_conv, self.final_channels, 7, padding=3, stride=2)
        #    self.batch4 = BatchNorm1d(self.final_channels)
        #self.conv1d_5 = torch.nn.Conv1d(self.channels_conv * 8, self.channels_conv * 16, 7, padding=3, stride=2)
        #self.batch5 = BatchNorm1d(self.channels_conv * 16)

        self.lin_temporal = nn.Linear(self.channels_conv * 8 * self.final_feature_size, self.TEMPORAL_EMBED_SIZE)

        # TCNs for temporal domain
        #self.tcn = TemporalConvNet(1, [self.channels_conv, self.channels_conv], kernel_size=7, stride=2,
        #                           dropout=self.dropout,
        #                           num_time_length=num_time_length / 2)

        if conv_strategy == ConvStrategy.TCN_2:
            self.temporal_conv = self.tcn
        elif conv_strategy == ConvStrategy.CNN_2:
            self.temporal_conv = nn.Sequential(self.conv1d_1, self.batch1, self.activation,
                                                     self.conv1d_2, self.batch2, self.activation,
                                                     self.conv1d_3, self.batch3, self.activation,
                                                     self.conv1d_4, self.batch4, self.activation)
        elif conv_strategy == ConvStrategy.ENTIRE:
            self.temporal_conv = nn.Sequential(self.conv1d_1, self.activation, self.batch1,
                                                     self.conv1d_2, self.activation, self.batch2,
                                                     self.conv1d_3, self.activation, self.batch3,
                                                     self.conv1d_4, self.activation, self.batch4)

        if self.pooling == PoolingStrategy.DIFFPOOL:
            self.final_linear = nn.Linear(ceil(self.TEMPORAL_EMBED_SIZE / 3), 1)
        else:
            self.final_linear = nn.Linear(self.TEMPORAL_EMBED_SIZE, 1)
        #self.final_linear.register_hook(set_grad(self.final_linear))

        self.diff_pool = DiffPoolLayer(272,
                                       self.TEMPORAL_EMBED_SIZE)

        self.init_weights()

    def init_weights(self):
        self.conv1d_1.weight.data.normal_(0, 0.01)
        self.conv1d_2.weight.data.normal_(0, 0.01)
        self.conv1d_3.weight.data.normal_(0, 0.01)
        self.conv1d_4.weight.data.normal_(0, 0.01)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        #x = x[:,:100]
        #print("OO", x.shape)

        # Temporal Convolutions
        if self.conv_strategy != ConvStrategy.ENTIRE:
            half_slice = int(self.num_time_length / 2)
            x_left = x[:, :half_slice].view(-1, 1, half_slice)
            x_right = x[:, half_slice:].view(-1, 1, half_slice)

            x_left = self.temporal_conv(x_left)
            x_right = self.temporal_conv(x_right)

            x = torch.cat([x_left, x_right], dim=1)

        elif self.conv_strategy == ConvStrategy.ENTIRE:
            x = x.view(-1, 1, self.num_time_length)
            #x = x.view(-1, 1, 100)
            x = self.temporal_conv(x)
            #print("1", x.shape)

        # Concatenating for the final embedding per node
        x = x.view(-1, self.channels_conv * 8 * self.final_feature_size)
        #print("1", x.shape)
        x = self.lin_temporal(x)
        #print("1", x.shape)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.add_gcn:
            x = self.gcn_conv1(x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, training=self.training)

        # x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # x = self.batch1(x)
        #print("befor", edge_index.shape, x.shape, data.batch.shape)
        adj_tmp = pyg_utils.to_dense_adj(edge_index, data.batch)
        #print("after", adj_tmp.shape)
        x_tmp, batch_mask = pyg_utils.to_dense_batch(x, data.batch)
        #print("after2", x_tmp.shape, batch_mask.shape)

        #res, lo, el = self.diff_pool(x_tmp, adj_tmp, batch_mask)
        #print("after_diff", res.shape, lo.shape, el.shape)

        if self.pooling == PoolingStrategy.MEAN:
            x = global_mean_pool(x, data.batch)
        elif self.pooling == PoolingStrategy.DIFFPOOL:
            x, link_loss, ent_loss = self.diff_pool(x_tmp, adj_tmp, batch_mask)
        #print("after_pool", x.shape)

        # elif self.pooling == 'mixed':
        #    x1 = global_max_pool(x, data.batch)
        #    x2 = global_add_pool(x, data.batch) <- maybe not this one?
        #    x3 = global_mean_pool(x, data.batch)
        #    x = torch.cat([x1, x2, x3], dim=1)
        # x = global_sort_pool(x, data.batch, 10)# esta linha funciona...
        # x, b = to_dense_batch(x, data.batch) # para numero de nós variáveis, isto não dá...
        # x = x.view(-1, x.shape[1] * 50)
        # Resulting shape of to_dense_batch is batch x num_nodes x num_features (at least for my regular graphs

        # x = F.dropout(x, p=dropout_perc, training=self.training)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.conv1d(x.unsqueeze(2)).squeeze(2)
        x = self.final_linear(x)
        # x = F.relu(x)
        # x = self.lin2(x)
        # TODO: try dense_diff_pool and DynamicEdgeConv
        if self.final_sigmoid:
            return torch.sigmoid(x) if self.pooling != PoolingStrategy.DIFFPOOL else (torch.sigmoid(x), link_loss, ent_loss)
        else:
            return x if self.pooling != PoolingStrategy.DIFFPOOL else (x, link_loss, ent_loss)

    def to_string_name(self):
        model_vars = ['D_' + str(self.dropout),
                      'A_' + self.activation_str,
                      'P_' + self.pooling.value,
                      'CS_' + self.conv_strategy.value,
                      'CHC_' + str(self.channels_conv),
                      'FS_' + str(self.final_sigmoid),
                      'GCN_' + str(self.add_gcn),
                      'GAT_' + str(self.add_gat)
                      ]

        return '__'.join(model_vars)
