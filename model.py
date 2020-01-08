from sys import exit

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch_geometric.nn import global_mean_pool, GCNConv

from utils import ConvStrategy

from torch.nn.utils import weight_norm
from tcn import TemporalConvNet

# function to extract grad
def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook

class SpatioTemporalModel(nn.Module):
    def __init__(self, num_time_length, dropout_perc, pooling, channels_conv, activation, conv_strategy,
                 add_gat=False, add_gcn=False, final_sigmoid=True):
        super(SpatioTemporalModel, self).__init__()

        if pooling != 'mean':
            print("THIS IS NOT PREPARED FOR OTHER POOLING THAN MEAN")
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

        self.final_linear = nn.Linear(self.TEMPORAL_EMBED_SIZE, 1)
        #self.final_linear.register_hook(set_grad(self.final_linear))

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

        if self.pooling == 'mean':
            x = global_mean_pool(x, data.batch)

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
        # return x
        if self.final_sigmoid:
            return torch.sigmoid(x)
        else:
            return x

    def to_string_name(self):
        model_vars = ['D_' + str(self.dropout),
                      'A_' + self.activation_str,
                      'P_' + self.pooling,
                      'CS_' + self.conv_strategy.value,
                      'CHC_' + str(self.channels_conv),
                      'FS_' + str(self.final_sigmoid),
                      'GCN_' + str(self.add_gcn),
                      'GAT_' + str(self.add_gat)
                      ]

        return '__'.join(model_vars)
