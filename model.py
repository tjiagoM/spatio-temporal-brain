from sys import exit
from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch_geometric.nn import global_mean_pool, GCNConv, GATConv
import torch_geometric.utils as pyg_utils
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from torch_geometric.utils import to_dense_batch

from utils import ConvStrategy, PoolingStrategy, EncodingStrategy

from torch.nn.utils import weight_norm
from tcn import TemporalConvNet

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
        self.INTERN_EMBED_SIZE = self.init_feats#ceil(self.init_feats / 3)

        num_nodes = ceil(0.25 * self.max_nodes)
        self.gnn1_pool = GNN(self.init_feats, self.INTERN_EMBED_SIZE, num_nodes, add_loop=True)
        self.gnn1_embed = GNN(self.init_feats, self.INTERN_EMBED_SIZE, self.INTERN_EMBED_SIZE, add_loop=True, lin=False)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(3 * self.INTERN_EMBED_SIZE, self.INTERN_EMBED_SIZE, num_nodes)
        self.gnn2_embed = GNN(3 * self.INTERN_EMBED_SIZE, self.INTERN_EMBED_SIZE, self.INTERN_EMBED_SIZE, lin=False)

        self.gnn3_embed = GNN(3 * self.INTERN_EMBED_SIZE, self.INTERN_EMBED_SIZE, self.INTERN_EMBED_SIZE, lin=False)

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
                 encoding_model=None, num_gnn_layers=1, gat_heads=0,
                 add_gat=False, add_gcn=False, final_sigmoid=True, num_nodes=None):
        super(SpatioTemporalModel, self).__init__()

        self.VERSION = '5.0'

        if pooling not in [PoolingStrategy.MEAN, PoolingStrategy.DIFFPOOL, PoolingStrategy.CONCAT]:
            print("THIS IS NOT PREPARED FOR OTHER POOLING THAN MEAN/DIFFPOOL/CONCAT")
            exit(-1)
        if conv_strategy not in [ConvStrategy.TCN_ENTIRE, ConvStrategy.CNN_ENTIRE]:
            print("THIS IS NOT PREPARED FOR OTHER CONV STRATEGY THAN ENTIRE/TCN_ENTIRE")
            exit(-1)
        if activation not in ['relu', 'tanh', 'elu']:
            print("THIS IS NOT PREPARED FOR OTHER ACTIVATION THAN relu/tanh/elu")
            exit(-1)
        if add_gat and add_gcn:
            print("You cannot have both GCN and GAT")
            exit(-1)

        if encoding_model is None:
            self.TEMPORAL_EMBED_SIZE = 256
            self.encoder_name = 'None'
        else:
            self.TEMPORAL_EMBED_SIZE = encoding_model.EMBED_SIZE
            self.encoder_name = encoding_model.MODEL_NAME
        self.encoder_model = encoding_model

        self.dropout = dropout_perc
        self.pooling = pooling
        dict_activations = {'relu': nn.ReLU(),
                            'elu': nn.ELU(),
                            'tanh': nn.Tanh()}
        self.activation = dict_activations[activation]
        self.activation_str = activation

        self.conv_strategy = conv_strategy
        self.num_nodes = num_nodes

        self.channels_conv = channels_conv
        self.final_channels = 1 if channels_conv == 1 else channels_conv * 2
        self.final_sigmoid = final_sigmoid
        self.add_gcn = add_gcn
        self.add_gat = add_gat  # TODO

        self.num_time_length = num_time_length
        self.final_feature_size = ceil(self.num_time_length / 2 / 8)

        self.num_gnn_layers = num_gnn_layers
        self.gat_heads = gat_heads
        if self.add_gcn:
            self.gnn_conv1 = GCNConv(self.TEMPORAL_EMBED_SIZE,
                                     self.TEMPORAL_EMBED_SIZE)
            if self.num_gnn_layers == 2:
                self.gnn_conv2 = GCNConv(self.TEMPORAL_EMBED_SIZE,
                                         self.TEMPORAL_EMBED_SIZE)
        elif self.add_gat:
            self.gnn_conv1 = GATConv(self.TEMPORAL_EMBED_SIZE,
                                     self.TEMPORAL_EMBED_SIZE,
                                     heads=self.gat_heads,
                                     concat=False,
                                     dropout=dropout_perc)
            if self.num_gnn_layers == 2:
                self.gnn_conv2 = GATConv(self.TEMPORAL_EMBED_SIZE,
                                         self.TEMPORAL_EMBED_SIZE,
                                         heads=self.gat_heads if self.gat_heads == 1 else int(self.gat_heads / 2),
                                         concat=False,
                                         dropout=dropout_perc)

        if self.encoder_model is not None:
            pass # Just it does not go to convolutions
        elif self.conv_strategy == ConvStrategy.TCN_ENTIRE:
            self.size_before_lin_temporal = self.channels_conv * 8 * self.final_feature_size
            self.lin_temporal = nn.Linear(self.size_before_lin_temporal, self.TEMPORAL_EMBED_SIZE)

            self.temporal_conv = TemporalConvNet(1,
                                                  [self.channels_conv, self.channels_conv * 2,
                                                   self.channels_conv * 4, self.channels_conv * 8],
                                                  kernel_size=7,
                                                  stride=2,
                                                  dropout=self.dropout,
                                                  num_time_length=self.num_time_length)
        elif self.conv_strategy == ConvStrategy.CNN_ENTIRE:# or self.conv_strategy == ConvStrategy.CNN_NO_STRIDES:
            stride = 2# if self.conv_strategy == ConvStrategy.CNN_ENTIRE else 1
            padding = 3# if self.conv_strategy == ConvStrategy.CNN_ENTIRE else 0
            self.size_before_lin_temporal = self.channels_conv * 8 * self.final_feature_size
            self.lin_temporal = nn.Linear(self.size_before_lin_temporal, self.TEMPORAL_EMBED_SIZE)

            self.conv1d_1 = nn.Conv1d(1, self.channels_conv, 7, padding=padding, stride=stride)
            self.conv1d_2 = nn.Conv1d(self.channels_conv, self.channels_conv * 2, 7, padding=padding, stride=stride)
            self.conv1d_3 = nn.Conv1d(self.channels_conv * 2, self.channels_conv * 4, 7, padding=padding, stride=stride)
            self.conv1d_4 = nn.Conv1d(self.channels_conv * 4, self.channels_conv * 8, 7, padding=padding, stride=stride)
            self.batch1 = BatchNorm1d(self.channels_conv)
            self.batch2 = BatchNorm1d(self.channels_conv * 2)
            self.batch3 = BatchNorm1d(self.channels_conv * 4)
            self.batch4 = BatchNorm1d(self.channels_conv * 8)

            #if self.conv_strategy == ConvStrategy.CNN_ENTIRE:
            self.temporal_conv = nn.Sequential(self.conv1d_1, self.activation, self.batch1, nn.Dropout(dropout_perc),
                                                 self.conv1d_2, self.activation, self.batch2, nn.Dropout(dropout_perc),
                                                 self.conv1d_3, self.activation, self.batch3, nn.Dropout(dropout_perc),
                                                 self.conv1d_4, self.activation, self.batch4, nn.Dropout(dropout_perc))
            #elif self.conv_strategy == ConvStrategy.CNN_NO_STRIDES:
            #    self.temporal_conv = nn.Sequential(self.conv1d_1, nn.MaxPool1d(2), self.activation, self.batch1, nn.Dropout(dropout_perc),
            #                                       self.conv1d_2, nn.MaxPool1d(2), self.activation, self.batch2, nn.Dropout(dropout_perc),
            #                                       self.conv1d_3, nn.MaxPool1d(2), self.activation, self.batch3, nn.Dropout(dropout_perc),
            #                                       self.conv1d_4, nn.MaxPool1d(2), self.activation, self.batch4, nn.Dropout(dropout_perc))

            self.init_weights()



        if self.pooling == PoolingStrategy.DIFFPOOL:
            self.pre_final_linear = nn.Linear(3 * self.TEMPORAL_EMBED_SIZE, self.TEMPORAL_EMBED_SIZE)

            self.diff_pool = DiffPoolLayer(num_nodes,
                                           self.TEMPORAL_EMBED_SIZE)
        elif self.pooling == PoolingStrategy.CONCAT:
            self.pre_final_linear = nn.Linear(self.num_nodes * self.TEMPORAL_EMBED_SIZE, self.TEMPORAL_EMBED_SIZE)

        self.final_linear = nn.Linear(self.TEMPORAL_EMBED_SIZE, 1)

    def init_weights(self):
        self.conv1d_1.weight.data.normal_(0, 0.01)
        self.conv1d_2.weight.data.normal_(0, 0.01)
        self.conv1d_3.weight.data.normal_(0, 0.01)
        self.conv1d_4.weight.data.normal_(0, 0.01)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Temporal Convolutions
        if self.encoder_model is None:
            x = x.view(-1, 1, self.num_time_length)
            x = self.temporal_conv(x)

            # Concatenating for the final embedding per node
            x = x.view(-1, self.size_before_lin_temporal)
            x = self.lin_temporal(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        elif self.encoder_name == EncodingStrategy.VAE3layers.value:
            mu, logvar = self.encoder_model.encode(x)
            x = self.encoder_model.reparameterize(mu, logvar)
        else:
            x = self.encoder_model.encode(x)

        if self.add_gcn or self.add_gat:
            x = self.gnn_conv1(x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, training=self.training)
            if self.num_gnn_layers == 2:
                x = self.gnn_conv2(x, edge_index)
                x = self.activation(x)
                x = F.dropout(x, training=self.training)

        if self.pooling == PoolingStrategy.MEAN:
            x = global_mean_pool(x, data.batch)
        elif self.pooling == PoolingStrategy.DIFFPOOL:
            adj_tmp = pyg_utils.to_dense_adj(edge_index, data.batch)
            x_tmp, batch_mask = pyg_utils.to_dense_batch(x, data.batch)

            x, link_loss, ent_loss = self.diff_pool(x_tmp, adj_tmp, batch_mask)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.activation(self.pre_final_linear(x))
        elif self.pooling == PoolingStrategy.CONCAT:
            x, _ = to_dense_batch(x, data.batch)
            x = x.view(-1, self.TEMPORAL_EMBED_SIZE * self.num_nodes)
            x = self.activation(self.pre_final_linear(x))

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final_linear(x)

        # TODO: try DynamicEdgeConv
        if self.final_sigmoid:
            return torch.sigmoid(x) if self.pooling != PoolingStrategy.DIFFPOOL else (torch.sigmoid(x), link_loss, ent_loss)
        else:
            return x if self.pooling != PoolingStrategy.DIFFPOOL else (x, link_loss, ent_loss)

    def to_string_name(self):
        model_vars = ['V_' + self.VERSION,
                      'TL_' + str(self.num_time_length),
                      'D_' + str(self.dropout),
                      'A_' + self.activation_str,
                      'P_' + self.pooling.value,
                      'CS_' + self.conv_strategy.value,
                      'CHC_' + str(self.channels_conv),
                      'FS_' + str(self.final_sigmoid),
                      'GCN_' + str(self.add_gcn),
                      'GAT_' + str(self.add_gat),
                      'GATH_' + str(self.gat_heads),
                      'NGNN_' + str(self.num_gnn_layers),
                      'ENC_' + str(self.encoder_name)
                      ]

        return '__'.join(model_vars)
