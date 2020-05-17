from sys import exit

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
from math import ceil
from torch.nn import BatchNorm1d
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from torch_geometric.nn import MetaLayer
from torch_geometric.nn import global_mean_pool, GCNConv, GATConv
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_mean

from tcn import TemporalConvNet
from utils import ConvStrategy, PoolingStrategy, EncodingStrategy, SweepType


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
        self.INTERN_EMBED_SIZE = self.init_feats  # ceil(self.init_feats / 3)

        num_nodes = ceil(0.25 * self.max_nodes)
        self.gnn1_pool = GNN(self.init_feats, self.INTERN_EMBED_SIZE, num_nodes, add_loop=True)
        self.gnn1_embed = GNN(self.init_feats, self.INTERN_EMBED_SIZE, self.INTERN_EMBED_SIZE, add_loop=True, lin=False)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(3 * self.INTERN_EMBED_SIZE, self.INTERN_EMBED_SIZE, num_nodes)
        self.gnn2_embed = GNN(3 * self.INTERN_EMBED_SIZE, self.INTERN_EMBED_SIZE, self.INTERN_EMBED_SIZE, lin=False)

        self.gnn3_embed = GNN(3 * self.INTERN_EMBED_SIZE, self.INTERN_EMBED_SIZE, self.INTERN_EMBED_SIZE, lin=False)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)

        return x, l1 + l2, e1 + e2


class EdgeModel(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features):
        super().__init__()
        self.input_size = 2 * num_node_features + num_edge_features
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.input_size, int(self.input_size / 2)),
            nn.ReLU(),
            nn.Linear(int(self.input_size / 2), num_edge_features),
        )

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr], 1)
        out = self.edge_mlp(out)
        return out


class NodeModel(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features):
        super(NodeModel, self).__init__()
        self.input_size = num_node_features + num_edge_features
        self.node_mlp_1 = nn.Sequential(
            nn.Linear(self.input_size, self.input_size * 2),
            nn.ReLU(),
            nn.Linear(self.input_size * 2, self.input_size * 2),
        )
        self.node_mlp_2 = nn.Sequential(
            nn.Linear(num_node_features + self.input_size * 2, self.input_size),
            nn.ReLU(),
            nn.Linear(self.input_size, num_node_features),
        )

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        # Scatter around "col" (destination nodes)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        # Concatenate X with transformed representation given the source nodes with edge's messages
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)


class SpatioTemporalModel(nn.Module):
    def __init__(self, num_time_length: int, dropout_perc: float, pooling: PoolingStrategy, channels_conv: int,
                 activation: str, conv_strategy: ConvStrategy, sweep_type: SweepType, num_gnn_layers: int = 1,
                 gat_heads: int = 0, multimodal_size: int = 0, temporal_embed_size: int = 16, model_version: str = '70',
                 encoding_strategy: EncodingStrategy = EncodingStrategy.NONE, encoding_model=None,
                 edge_weights: bool = False, final_sigmoid: bool = True, num_nodes: int = None):
        super(SpatioTemporalModel, self).__init__()

        self.VERSION = model_version

        if pooling not in [PoolingStrategy.MEAN, PoolingStrategy.DIFFPOOL, PoolingStrategy.CONCAT]:
            print('THIS IS NOT PREPARED FOR OTHER POOLING THAN MEAN/DIFFPOOL/CONCAT')
            exit(-1)
        if conv_strategy not in [ConvStrategy.TCN_ENTIRE, ConvStrategy.CNN_ENTIRE, ConvStrategy.NONE]:
            print('THIS IS NOT PREPARED FOR OTHER CONV STRATEGY THAN ENTIRE/TCN_ENTIRE/NONE')
            exit(-1)
        if activation not in ['relu', 'tanh', 'elu']:
            print('THIS IS NOT PREPARED FOR OTHER ACTIVATION THAN relu/tanh/elu')
            exit(-1)
        if sweep_type == SweepType.GAT:
            print('GAT is not ready for edge_attr')
            exit(-1)
        if conv_strategy != ConvStrategy.NONE and encoding_strategy not in [EncodingStrategy.NONE,
                                                                            EncodingStrategy.STATS]:
            print('Mismatch on conv_strategy/encoding_strategy')
            exit(-1)

        self.multimodal_size: int = multimodal_size
        self.TEMPORAL_EMBED_SIZE: int = temporal_embed_size
        self.NODE_EMBED_SIZE: int = self.TEMPORAL_EMBED_SIZE + self.multimodal_size

        if self.multimodal_size > 0:
            self.multimodal_lin = nn.Linear(self.multimodal_size, self.multimodal_size)
            self.multimodal_batch = BatchNorm1d(self.multimodal_size)

        self.encoding_strategy = encoding_strategy
        self.encoder_model = encoding_model
        if encoding_model is not None:
            self.NODE_EMBED_SIZE = self.encoding_model.EMBED_SIZE

        if self.encoding_strategy == EncodingStrategy.STATS:
            self.stats_lin = nn.Linear(self.TEMPORAL_EMBED_SIZE, self.TEMPORAL_EMBED_SIZE)
            self.stats_batch = BatchNorm1d(self.TEMPORAL_EMBED_SIZE)

        self.dropout: float = dropout_perc
        self.pooling = pooling
        dict_activations = {'relu': nn.ReLU(),
                            'elu': nn.ELU(),
                            'tanh': nn.Tanh()}
        self.activation = dict_activations[activation]
        self.activation_str = activation

        self.conv_strategy = conv_strategy
        self.num_nodes = num_nodes

        self.channels_conv = channels_conv
        self.final_sigmoid = final_sigmoid
        self.sweep_type = sweep_type

        self.num_time_length = num_time_length
        self.final_feature_size = ceil(self.num_time_length / 2 / 8)

        self.edge_weights = edge_weights
        self.num_gnn_layers = num_gnn_layers
        self.gat_heads = gat_heads
        if self.sweep_type == SweepType.GCN:
            self.gnn_conv1 = GCNConv(self.NODE_EMBED_SIZE,
                                     self.NODE_EMBED_SIZE)
            if self.num_gnn_layers == 2:
                self.gnn_conv2 = GCNConv(self.NODE_EMBED_SIZE,
                                         self.NODE_EMBED_SIZE)
        elif self.sweep_type == SweepType.GAT:
            self.gnn_conv1 = GATConv(self.NODE_EMBED_SIZE,
                                     self.NODE_EMBED_SIZE,
                                     heads=self.gat_heads,
                                     concat=False,
                                     dropout=dropout_perc)
            if self.num_gnn_layers == 2:
                self.gnn_conv2 = GATConv(self.NODE_EMBED_SIZE,
                                         self.NODE_EMBED_SIZE,
                                         heads=self.gat_heads if self.gat_heads == 1 else int(self.gat_heads / 2),
                                         concat=False,
                                         dropout=dropout_perc)
        elif self.sweep_type == SweepType.META_EDGE_NODE:
            self.meta_layer = MetaLayer(edge_model=EdgeModel(num_node_features=self.NODE_EMBED_SIZE,
                                                             num_edge_features=1),
                                        node_model=NodeModel(num_node_features=self.NODE_EMBED_SIZE,
                                                             num_edge_features=1))
        elif self.sweep_type == SweepType.META_NODE:
            self.meta_layer = MetaLayer(node_model=NodeModel(num_node_features=self.NODE_EMBED_SIZE,
                                                             num_edge_features=1))

        if self.conv_strategy == ConvStrategy.TCN_ENTIRE:
            self.size_before_lin_temporal = self.channels_conv * 8 * self.final_feature_size
            self.lin_temporal = nn.Linear(self.size_before_lin_temporal, self.NODE_EMBED_SIZE - self.multimodal_size)

            self.temporal_conv = TemporalConvNet(1,
                                                 [self.channels_conv, self.channels_conv * 2,
                                                  self.channels_conv * 4, self.channels_conv * 8],
                                                 kernel_size=7,
                                                 stride=2,
                                                 dropout=self.dropout,
                                                 num_time_length=self.num_time_length)
        elif self.conv_strategy == ConvStrategy.CNN_ENTIRE:
            stride = 2
            padding = 3
            self.size_before_lin_temporal = self.channels_conv * 8 * self.final_feature_size
            self.lin_temporal = nn.Linear(self.size_before_lin_temporal, self.NODE_EMBED_SIZE - self.multimodal_size)

            self.conv1d_1 = nn.Conv1d(1, self.channels_conv, 7, padding=padding, stride=stride)
            self.conv1d_2 = nn.Conv1d(self.channels_conv, self.channels_conv * 2, 7, padding=padding, stride=stride)
            self.conv1d_3 = nn.Conv1d(self.channels_conv * 2, self.channels_conv * 4, 7, padding=padding, stride=stride)
            self.conv1d_4 = nn.Conv1d(self.channels_conv * 4, self.channels_conv * 8, 7, padding=padding, stride=stride)
            self.batch1 = BatchNorm1d(self.channels_conv)
            self.batch2 = BatchNorm1d(self.channels_conv * 2)
            self.batch3 = BatchNorm1d(self.channels_conv * 4)
            self.batch4 = BatchNorm1d(self.channels_conv * 8)

            self.temporal_conv = nn.Sequential(self.conv1d_1, self.activation, self.batch1, nn.Dropout(dropout_perc),
                                               self.conv1d_2, self.activation, self.batch2, nn.Dropout(dropout_perc),
                                               self.conv1d_3, self.activation, self.batch3, nn.Dropout(dropout_perc),
                                               self.conv1d_4, self.activation, self.batch4, nn.Dropout(dropout_perc))

            self.init_weights()

        if self.pooling == PoolingStrategy.DIFFPOOL:
            self.pre_final_linear = nn.Linear(3 * self.NODE_EMBED_SIZE, self.NODE_EMBED_SIZE)

            self.diff_pool = DiffPoolLayer(num_nodes,
                                           self.NODE_EMBED_SIZE)
        elif self.pooling == PoolingStrategy.CONCAT:
            self.pre_final_linear = nn.Linear(self.num_nodes * self.NODE_EMBED_SIZE, self.NODE_EMBED_SIZE)

        self.final_linear = nn.Linear(self.NODE_EMBED_SIZE, 1)

    def init_weights(self):
        self.conv1d_1.weight.data.normal_(0, 0.01)
        self.conv1d_2.weight.data.normal_(0, 0.01)
        self.conv1d_3.weight.data.normal_(0, 0.01)
        self.conv1d_4.weight.data.normal_(0, 0.01)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        if self.multimodal_size > 0:
            xn, x = x[:, :self.multimodal_size], x[:, self.multimodal_size:]
            xn = self.multimodal_lin(xn)
            xn = self.activation(xn)
            xn = self.multimodal_batch(xn)
            xn = F.dropout(xn, p=self.dropout, training=self.training)

        # Processing temporal part
        if self.conv_strategy != ConvStrategy.NONE:
            x = x.view(-1, 1, self.num_time_length)
            x = self.temporal_conv(x)

            # Concatenating for the final embedding per node
            x = x.view(-1, self.size_before_lin_temporal)
            x = self.lin_temporal(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        elif self.encoding_strategy == EncodingStrategy.STATS:
            x = self.stats_lin(x)
            x = self.activation(x)
            x = self.stats_batch(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        elif self.encoding_strategy == EncodingStrategy.VAE3layers:
            mu, logvar = self.encoder_model.encode(x)
            x = self.encoder_model.reparameterize(mu, logvar)
        elif self.encoding_strategy == EncodingStrategy.AE3layers:
            x = self.encoder_model.encode(x)

        if self.multimodal_size > 0:
            x = torch.cat((xn, x), dim=1)

        if self.sweep_type in [SweepType.GAT, SweepType.GCN]:
            if self.edge_weights:
                x = self.gnn_conv1(x, edge_index, edge_weight=edge_attr.view(-1))
            else:
                x = self.gnn_conv1(x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, training=self.training)
            if self.num_gnn_layers == 2:
                if self.edge_weights:
                    x = self.gnn_conv2(x, edge_index, edge_weight=edge_attr.view(-1))
                else:
                    x = self.gnn_conv2(x, edge_index)
                x = self.activation(x)
                x = F.dropout(x, training=self.training)
        elif self.sweep_type in [SweepType.META_NODE, SweepType.META_EDGE_NODE]:
            x, edge_attr, _ = self.meta_layer(x, edge_index, edge_attr)

        if self.pooling == PoolingStrategy.MEAN:
            x = global_mean_pool(x, data.batch)
        elif self.pooling == PoolingStrategy.DIFFPOOL:
            adj_tmp = pyg_utils.to_dense_adj(edge_index, data.batch, edge_attr=edge_attr)
            if edge_attr is not None: # Because edge_attr only has 1 feature per edge
                adj_tmp = adj_tmp[:, :, :, 0]
            x_tmp, batch_mask = pyg_utils.to_dense_batch(x, data.batch)

            x, link_loss, ent_loss = self.diff_pool(x_tmp, adj_tmp, batch_mask)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.activation(self.pre_final_linear(x))
        elif self.pooling == PoolingStrategy.CONCAT:
            x, _ = to_dense_batch(x, data.batch)
            x = x.view(-1, self.NODE_EMBED_SIZE * self.num_nodes)
            x = self.activation(self.pre_final_linear(x))

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final_linear(x)

        if self.final_sigmoid:
            return torch.sigmoid(x) if self.pooling != PoolingStrategy.DIFFPOOL else (
                torch.sigmoid(x), link_loss, ent_loss)
        else:
            return x if self.pooling != PoolingStrategy.DIFFPOOL else (x, link_loss, ent_loss)

    def to_string_name(self):
        model_vars = ['V_' + self.VERSION,
                      'TL_' + str(self.num_time_length),
                      'D_' + str(self.dropout),
                      'A_' + self.activation_str,
                      'P_' + self.pooling.value[:3],
                      'CS_' + self.conv_strategy.value[:3],
                      'CH_' + str(self.channels_conv),
                      'FS_' + str(self.final_sigmoid)[:1],
                      'T_' + self.sweep_type.value[:3],
                      'W_' + str(self.edge_weights)[:1],
                      'GH_' + str(self.gat_heads),
                      'GL_' + str(self.num_gnn_layers),
                      'E_' + self.encoding_strategy.value[:3],
                      'M_' + str(self.multimodal_size),
                      'S_' + str(self.TEMPORAL_EMBED_SIZE)
                      ]

        return ''.join(model_vars)
