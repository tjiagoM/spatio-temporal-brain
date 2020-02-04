import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch_geometric.data import InMemoryDataset, Data

from numpy.random import default_rng

from utils import NEW_STRUCT_PEOPLE, NEW_MULTIMODAL_TIMESERIES, Normalisation, ConnType, get_timeseries_final_path, \
    OLD_NETMATS_PEOPLE

PEOPLE_DEMOGRAPHICS_PATH = 'meta_data/people_demographics.csv'


def get_struct_path(person):
    return f'../hcp_multimodal_parcellation/HCP_tracks_matrices_BN_withcerebellum/{person}/{person}_{person}_BN_Atlas_246_1mm_geom_withcerebellum_RS.txt'

def get_adj_50_path(person, index):
    return f'../../../space/hcp_50_timeseries/processed_4_split_50/{person}_{index}.npy'

def get_50_ts_path(person):
    return f'../hcp_timecourses/3T_HCP1200_MSMAll_d50_ts2/{person}.txt'

class HCPDataset(InMemoryDataset):
    def __init__(self, root, target_var, num_nodes, threshold, connectivity_type, normalisation, disconnect_nodes=False,
                 transform=None, pre_transform=None):
        '''

        :param root:
        :param target_var:
        :param num_nodes:
        :param threshold: Between [0, 100]
        :param connectivity_type:
        :param transform:
        :param pre_transform:
        '''
        if target_var not in ['gender']:
            print("NOT A VALID target_var!")
            exit(-2)
        if threshold < 0 or threshold > 100:
            print("NOT A VALID threshold!")
            exit(-2)
        if connectivity_type not in [ConnType.FMRI, ConnType.STRUCT]:
            print("NOT A VALID connectivity_type!")
            exit(-2)
        if normalisation not in [Normalisation.NONE, Normalisation.ROI, Normalisation.SUBJECT]:
            print("NOT A VALID normalisation!")
            exit(-2)

        # TODO: check whether this matches the name inside root
        self.target_var = target_var
        self.num_nodes = num_nodes
        self.threshold = threshold
        self.connectivity_type = connectivity_type
        self.disconnect_nodes = disconnect_nodes
        self.normalisation = normalisation

        if self.disconnect_nodes:
            print("Warning: Removing disconnected nodes not yet developed in HCPDataset")

        super(HCPDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data_hcp_func.dataset']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def __normalise_timeseries(self, timeseries):
        if self.normalisation == Normalisation.ROI:
            scaler = RobustScaler().fit(timeseries)
            timeseries = scaler.transform(timeseries).T
        elif self.normalisation == Normalisation.SUBJECT:
            flatten_timeseries = timeseries.flatten().reshape(-1, 1)
            scaler = RobustScaler().fit(flatten_timeseries)
            timeseries = scaler.transform(flatten_timeseries).reshape(timeseries.shape).T
        else:  # No normalisation
            timeseries = timeseries.T

        return timeseries

    def __create_thresholded_graph(self, adj_array):
        num_to_filter = int((self.threshold / 100.0) * (self.num_nodes * (self.num_nodes - 1) / 2))

        # Following code is similar to bctpy
        indices = np.where(adj_array)
        sorted_indices = np.argsort(adj_array[indices])[::-1]
        adj_array[(indices[0][sorted_indices][num_to_filter:], indices[1][sorted_indices][num_to_filter:])] = 0

        # Just to get a symmetrical matrix
        adj_array = adj_array + adj_array.T

        # Diagonals need connection of 1 for graph operations
        adj_array[np.diag_indices(self.num_nodes)] = 1.0

        return nx.from_numpy_array(adj_array, create_using=nx.DiGraph)

    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        # No sorted needed?
        if self.num_nodes == 50:
            filtered_people = OLD_NETMATS_PEOPLE
        else:
            filtered_people = sorted(list(set(NEW_MULTIMODAL_TIMESERIES).intersection(set(NEW_STRUCT_PEOPLE))))

        info_df = pd.read_csv(PEOPLE_DEMOGRAPHICS_PATH).set_index('Subject')

        ##########
        for person in filtered_people:
            if self.connectivity_type == ConnType.FMRI:
                if self.num_nodes != 50:
                    print("ConnType.FMRI not ready for num_nodes != 50")
                    exit(-2)

                all_ts = np.genfromtxt(get_50_ts_path(person))
                t1 = all_ts[:1200, :]
                t2 = all_ts[1200:2400, :]
                t3 = all_ts[2400:3600, :]
                t4 = all_ts[3600:, :]

                for ind, timeseries in enumerate([t1, t2, t3, t4]):
                    corr_arr = np.load(get_adj_50_path(person, ind))
                    # For threshold operations, zero out lower triangle (including diagonal)
                    corr_arr[np.tril_indices(self.num_nodes)] = 0

                    G = self.__create_thresholded_graph(corr_arr)

                    edge_index = torch.tensor(np.array(G.edges()), dtype=torch.long).t().contiguous()

                    timeseries = self.__normalise_timeseries(timeseries)
                    x = torch.tensor(timeseries, dtype=torch.float)
                    if self.target_var == 'gender':
                        y = torch.tensor([info_df.loc[person, 'Gender']], dtype=torch.float)
                    data = Data(x=x, edge_index=edge_index, y=y)  # edge_attr=edge_attr,
                    data.hcp_id = torch.tensor([person])
                    data.session = torch.tensor([1 if ind < 2 else 2])
                    data.direction = torch.tensor([0 if ind in [1, 3] else 1])
                    data_list.append(data)

            elif self.connectivity_type == ConnType.STRUCT:
                # arr_struct will only have values in the upper triangle
                arr_struct = np.genfromtxt(get_struct_path(person))

                G = self.__create_thresholded_graph(arr_struct)

                for session_day in [1, 2]:
                    edge_index = torch.tensor(np.array(G.edges()), dtype=torch.long).t().contiguous()

                    try:
                        path_lr, path_rl = get_timeseries_final_path(person, session_day, direction=True)
                        timeseries_lr = np.load(path_lr).T
                        timeseries_rl = np.load(path_rl).T
                        #timeseries_lr = np.random.normal(0, 0.1, (100, 20))
                        #timeseries_rl = np.random.normal(0, 0.1, (100, 20))

                    except FileNotFoundError:
                        print('W: No', person, session_day)
                        continue

                    for direction, timeseries in enumerate([timeseries_lr, timeseries_rl]):
                        timeseries = self.__normalise_timeseries(timeseries)

                        x = torch.tensor(timeseries, dtype=torch.float)  # torch.ones(50).unsqueeze(1)
                        #x = x[[16, 165, 80, 46, 56, 133, 237, 171, 230, 8, 36, 191, 199, 13, 3, 17, 149, 59, 53, 115],:]

                        if self.target_var == 'gender':
                            y = torch.tensor([info_df.loc[person, 'Gender']], dtype=torch.float)
                        else:
                            y = torch.tensor([0], dtype=torch.float)  # will be set later

                        # edge_attr = torch.tensor(list(nx.get_edge_attributes(G, 'weight').values()),
                        #                         dtype=torch.float).unsqueeze(1)

                        data = Data(x=x, edge_index=edge_index, y=y)  # edge_attr=edge_attr,
                        data.hcp_id = torch.tensor([person])
                        data.session = torch.tensor([session_day])
                        data.direction = torch.tensor([direction])
                        data_list.append(data)

        negative_num = len(list(filter(lambda x: x.y == 0, data_list)))
        positive_num = len(list(filter(lambda x: x.y == 1, data_list)))
        print("Negative class:", negative_num)
        print("Positive class:", positive_num)

        smallest_value = 1 if positive_num < negative_num else 0
        highest_value = 0 if positive_num < negative_num else 1

        if positive_num > negative_num:
            negative_num, positive_num = positive_num, negative_num


        # Randomly undersampling
        rng = default_rng(seed=0)
        numbers_sample = rng.choice(negative_num, size=positive_num, replace=False)

        y_0 = list(filter(lambda x: x.y == highest_value, data_list))
        data_list = list(filter(lambda x: x.y == smallest_value, data_list))
        y_0 = [elem for id, elem in enumerate(y_0) if id in numbers_sample]
        data_list.extend(y_0)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



def create_hcp_correlation_vals(num_nodes=50):
    final_dict = {}

    for person in OLD_NETMATS_PEOPLE:
        for ind in range(4):
            corr_arr = np.load(get_adj_50_path(person, ind))
            # For threshold operations, zero out lower triangle (including diagonal)
            flatten_array = corr_arr[np.triu_indices(num_nodes, k=1)]

            session = 1 if ind < 2 else 2
            direction = 0 if ind in [1, 3] else 1

            final_dict[(person, session, direction)] = flatten_array

    return final_dict