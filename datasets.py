import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch_geometric.data import InMemoryDataset, Data

from utils import NEW_STRUCT_PEOPLE, NEW_MULTIMODAL_TIMESERIES

PEOPLE_DEMOGRAPHICS_PATH = 'meta_data/people_demographics.csv'


def get_struct_path(person):
    return f'../hcp_multimodal_parcellation/HCP_tracks_matrices_BN_withcerebellum/{person}/{person}_{person}_BN_Atlas_246_1mm_geom_withcerebellum_RS.txt'


def get_timeseries_path(person, session_day):
    return f'../hcp_multimodal_parcellation/concatenated_timeseries/{person}_{session_day}.npy'


# TODO: Meter no construtor o numero de nós/type para guardar grafos diferentes
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
        if target_var not in ['gender', 'intelligence']:
            print("NOT A VALID target_var!")
            exit(-2)
        if threshold < 0 or threshold > 100:
            print("NOT A VALID threshold!")
            exit(-2)
        if connectivity_type not in ['fmri', 'struct']:
            print("NOT A VALID connectivity_type!")
            exit(-2)
        if normalisation not in ['no_norm', 'roi_norm', 'subject_norm']:
            print("NOT A VALID normalisation!")
            exit(-2)

        # TODO: check whether this matches the name inside root
        self.target_var = target_var
        self.num_nodes = num_nodes
        self.threshold = threshold
        self.connectivity_type = connectivity_type
        self.disconnect_nodes = disconnect_nodes
        self.normalisation = normalisation

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

    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        # No sorted needed?
        filtered_people = sorted(list(set(NEW_MULTIMODAL_TIMESERIES).intersection(set(NEW_STRUCT_PEOPLE))))

        info_df = pd.read_csv(PEOPLE_DEMOGRAPHICS_PATH).set_index('Subject')
        ##########
        for person in filtered_people:

            if self.connectivity_type == 'fmri':
                pass
            elif self.connectivity_type == 'struct':
                # arr_struct will only have values in the upper triangle
                arr_struct = np.genfromtxt(get_struct_path(person))
                num_to_filter = int((self.threshold / 100.0) * (self.num_nodes * (self.num_nodes - 1) / 2))

                # Following code is similar to bctpy
                indices = np.where(arr_struct)
                sorted_indices = np.argsort(arr_struct[indices])[::-1]
                arr_struct[(indices[0][sorted_indices][num_to_filter:], indices[1][sorted_indices][num_to_filter:])] = 0

                # Just to get a symmetrical matrix
                arr_struct = arr_struct + arr_struct.T

                # Diagonals need connection of 1 for graph operations
                arr_struct[np.diag_indices(self.num_nodes)] = 1.0

                G = nx.from_numpy_array(arr_struct, create_using=nx.DiGraph)

            if self.disconnect_nodes:
                print("Warning: Not yet developed in HCPDataset")
                pass

            for session_day in [1, 2]:

                edge_index = torch.tensor(np.array(G.edges()), dtype=torch.long).t().contiguous()

                try:
                    timeseries = np.load(get_timeseries_path(person, session_day)).T
                except FileNotFoundError:
                    print('W: No', person, session_day)
                    continue

                if self.normalisation == 'roi_norm':
                    scaler = RobustScaler().fit(timeseries)
                    timeseries = scaler.transform(timeseries).T

                x = torch.tensor(timeseries, dtype=torch.float)  # torch.ones(50).unsqueeze(1)

                if self.target_var == 'gender':
                    y = torch.tensor([info_df.loc[person, 'Gender']], dtype=torch.float)
                else:
                    y = torch.tensor([0], dtype=torch.float)  # will be set later

                # edge_attr = torch.tensor(list(nx.get_edge_attributes(G, 'weight').values()),
                #                         dtype=torch.float).unsqueeze(1)

                data = Data(x=x, edge_index=edge_index, y=y)  # edge_attr=edge_attr,
                data.hcp_id = torch.tensor([person])
                data.session = torch.tensor([session_day])
                data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
