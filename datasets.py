import networkx as nx
import numpy as np
import pandas as pd
import torch
import os
from sklearn.preprocessing import RobustScaler
from torch_geometric.data import InMemoryDataset, Data

from numpy.random import default_rng

from utils import DESIKAN_COMPLETE_TS, DESIKAN_TRACKS, Normalisation, ConnType, \
    OLD_NETMATS_PEOPLE, UKB_IDS_PATH, UKB_ADJ_ARR_PATH, UKB_TIMESERIES_PATH, UKB_PHENOTYPE_PATH

PEOPLE_DEMOGRAPHICS_PATH = 'meta_data/people_demographics.csv'


def get_adj_50_path(person, index, ts_split):
    return f'../../../space/hcp_50_timeseries/processed_{ts_split}_split_50/{person}_{index}.npy'

def get_50_ts_path(person):
    return f'../hcp_timecourses/3T_HCP1200_MSMAll_d50_ts2/{person}.txt'

def get_desikan_tracks_path(person):
    return f'/space/desikan_tracks/{person}/{person}_conn_aparc+aseg_RS_sl.txt'

def threshold_adj_array(adj_array, threshold, num_nodes):
    num_to_filter = int((threshold / 100.0) * (num_nodes * (num_nodes - 1) / 2))

    # For threshold operations, zero out lower triangle (including diagonal)
    adj_array[np.tril_indices(num_nodes)] = 0

    # Following code is similar to bctpy
    indices = np.where(adj_array)
    sorted_indices = np.argsort(adj_array[indices])[::-1]
    adj_array[(indices[0][sorted_indices][num_to_filter:], indices[1][sorted_indices][num_to_filter:])] = 0

    # Just to get a symmetrical matrix
    adj_array = adj_array + adj_array.T

    # Diagonals need connection of 1 for graph operations
    adj_array[np.diag_indices(num_nodes)] = 1.0

    return adj_array

class BrainDataset(InMemoryDataset):
    def __init__(self, root, target_var, num_nodes, threshold, connectivity_type, normalisation, time_length=1200,
                 disconnect_nodes=False, transform=None, pre_transform=None):
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
        self.time_length = time_length

        self.ts_split_num = int(4800 / time_length)

        if self.disconnect_nodes:
            print("Warning: Removing disconnected nodes not yet developed in HCPDataset")

        super(BrainDataset, self).__init__(root, transform, pre_transform)
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

        adj_array = threshold_adj_array(adj_array, self.threshold, self.num_nodes)

        return nx.from_numpy_array(adj_array, create_using=nx.DiGraph)

    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        # No sorted needed?
        if self.num_nodes == 50:
            filtered_people = OLD_NETMATS_PEOPLE
        # UK BIOBANK PEOPLE!
        elif self.num_nodes == 376:
            filtered_people = np.load(UKB_IDS_PATH) # start simple for comparison
        else: # multimodal part
            filtered_people = sorted(list(set(DESIKAN_COMPLETE_TS).intersection(set(DESIKAN_TRACKS))))

        if self.num_nodes == 376:
            info_df = pd.read_csv(UKB_PHENOTYPE_PATH, delimiter=',').set_index('eid')['31-0.0']
            info_df = info_df.apply(lambda x: 1 if x == 'Male' else 0)
        else:
            info_df = pd.read_csv(PEOPLE_DEMOGRAPHICS_PATH).set_index('Subject')

        ##########
        for person in filtered_people:
            # UK Biobank
            if self.num_nodes == 376:
                ts = np.loadtxt(f'{UKB_TIMESERIES_PATH}/UKB{person}_ts_raw.txt', delimiter=',')
                if ts.shape[1] == 523:
                    ts = ts[:, :490]
                # For normalisation part
                ts = ts.T

                corr_arr = np.load(f'{UKB_ADJ_ARR_PATH}/{person}.npy')
                G = self.__create_thresholded_graph(corr_arr)
                edge_index = torch.tensor(np.array(G.edges()), dtype=torch.long).t().contiguous()

                timeseries = self.__normalise_timeseries(ts)
                x = torch.tensor(timeseries, dtype=torch.float)

                if self.target_var == 'gender':
                    y = torch.tensor([info_df.loc[person]], dtype=torch.float)
                data = Data(x=x, edge_index=edge_index, y=y)
                data.ukb_id = torch.tensor([person])
                data_list.append(data)

            elif self.connectivity_type == ConnType.FMRI:
                if self.num_nodes not in [50]:
                    print("ConnType.FMRI not ready for num_nodes != 50")
                    exit(-2)

                all_ts = np.genfromtxt(get_50_ts_path(person))

                for ind, slice_start in enumerate(range(0, 4800, self.time_length)):
                    ts = all_ts[slice_start:slice_start + self.time_length, :]

                    corr_arr = np.load(get_adj_50_path(person, ind, ts_split=self.ts_split_num))

                    G = self.__create_thresholded_graph(corr_arr)

                    edge_index = torch.tensor(np.array(G.edges()), dtype=torch.long).t().contiguous()

                    timeseries = self.__normalise_timeseries(ts)
                    x = torch.tensor(timeseries, dtype=torch.float)
                    if self.target_var == 'gender':
                        y = torch.tensor([info_df.loc[person, 'Gender']], dtype=torch.float)
                    data = Data(x=x, edge_index=edge_index, y=y)  # edge_attr=edge_attr,
                    data.hcp_id = torch.tensor([person])
                    data.index = torch.tensor([ind])
                    #data.session = torch.tensor([1 if ind < 2 else 2])
                    #data.direction = torch.tensor([0 if ind in [1, 3] else 1])
                    data_list.append(data)

            elif self.connectivity_type == ConnType.STRUCT:
                # TODO: warning if time_length is not 1200
                # arr_struct will only have values in the upper triangle
                arr_struct = np.genfromtxt(get_desikan_tracks_path(person))

                G = self.__create_thresholded_graph(arr_struct)

                for person in filtered_people:
                    for ind in ['1_LR', '1_RL', '2_LR', '2_RL']:
                        ts = np.genfromtxt(
                            f'/space/desikan_timeseries/{person}_{ind}/{person}_rfMRI_REST{ind}_rfMRI_REST{ind}_hp2000_clean_T1_2_MNI2mm_shadowreg_aparc+aseg_nodes.txt')

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



def create_ukb_corrs_flatten(num_nodes=376):
    final_dict = {}

    for person in np.load(UKB_IDS_PATH):
        corr_arr = np.load(f'{UKB_ADJ_ARR_PATH}/{person}.npy')

        # Getting upper triangle only (without diagonal)
        flatten_array = corr_arr[np.triu_indices(num_nodes, k=1)]

        final_dict[person] = flatten_array

    return final_dict

def create_hcp_correlation_vals(num_nodes=50, ts_split_num=64, binarise=False, threshold=100):
    final_dict = {}

    for person in OLD_NETMATS_PEOPLE:
        for ind in range(ts_split_num):
            corr_arr = np.load(get_adj_50_path(person, ind, ts_split=ts_split_num))

            if binarise:
                corr_arr = threshold_adj_array(corr_arr, threshold, num_nodes)
            # Getting upper triangle only (without diagonal)
            flatten_array = corr_arr[np.triu_indices(num_nodes, k=1)]

            if binarise:
                flatten_array[flatten_array != 0] = 1

            final_dict[(person, ind)] = flatten_array

    return final_dict