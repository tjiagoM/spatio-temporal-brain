import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch_geometric.nn import global_mean_pool, GCNConv


class NetG(torch.nn.Module):
    def __init__(self, num_time_length, dropout_perc, pooling, channels_conv, activation, conv_strategy,
                 add_gat=False, add_gcn=False, final_sigmoid=True):
        super(NetG, self).__init__()

        if pooling != 'mean':
            print("THIS IS NOT PREPARED FOR OTHER POOLING THAN MEAN")
            exit(-1)
        if conv_strategy != 'entire':
            print("THIS IS NOT PREPARED FOR OTHER CONV STRATEGY THAN ENTIRE")
            exit(-1)
        if activation not in ['relu', 'tanh']:
            print("THIS IS NOT PREPARED FOR OTHER ACTIVATION THAN relu/tanh")
            exit(-1)

        self.dropout = dropout_perc
        self.pooling = pooling
        self.activation = F.relu if activation == 'relu' else F.tanh
        self.activation_str = activation

        self.conv_strategy = conv_strategy  # TODO

        self.channels_conv = channels_conv
        self.final_channels = 1 if channels_conv == 1 else channels_conv * 2
        self.final_sigmoid = final_sigmoid
        self.add_gcn = add_gcn
        self.add_gat = add_gat  # TODO

        # After convolutions/maxpoolings
        self.num_time_length = num_time_length
        self.final_feature_size = int(self.num_time_length / 16)

        self.gcn_conv1 = GCNConv(self.final_feature_size * self.final_channels,
                                 self.final_feature_size * self.final_channels)

        self.conv1d_1 = torch.nn.Conv1d(1, self.channels_conv, 3, padding=1, stride=1)
        self.maxpool1d_1 = torch.nn.MaxPool1d(4)

        self.conv1d_2 = torch.nn.Conv1d(self.channels_conv, self.final_channels, 5, padding=2,
                                        stride=1)
        self.maxpool1d_2 = torch.nn.MaxPool1d(4)

        self.batch1 = BatchNorm1d(self.final_feature_size * self.final_channels)
        self.lin1 = torch.nn.Linear(self.final_feature_size * self.final_channels, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # x = self.batch1(x)
        # BS/num_nodes X num_edges
        x = x.view(-1, 1, self.num_time_length)
        x = self.conv1d_1(x)  # smooth
        x = self.maxpool1d_1(x)
        x = self.conv1d_2(x)  # reduce
        x = self.maxpool1d_2(x)
        x = x.view(-1, self.final_feature_size * self.final_channels)
        # x = self.conv1(x, edge_index)
        x = self.batch1(x)
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
        x = self.lin1(x)
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
                      'CS_' + self.conv_strategy,
                      'CHC_' + str(self.channels_conv),
                      'FS_' + str(self.final_sigmoid),
                      'GCN_' + str(self.add_gcn),
                      'GAT_' + str(self.add_gat)
                      ]

        return '__'.join(model_vars)
