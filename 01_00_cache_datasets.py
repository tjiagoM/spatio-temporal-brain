from sklearn.model_selection import ParameterGrid

from datasets import HCPDataset
from utils import create_name_for_hcp_dataset, Normalisation, ConnType

if __name__ == '__main__':
    combinations = {'num_nodes': [272],
                    'target_var': ['gender'],
                    'threshold': [5, 10, 20],
                    'connectivity_type': ['struct'],
                    'normalisation': ['no_norm', 'roi_norm', 'subject_norm']}
    grid = ParameterGrid(combinations)

    for p in grid:
        name_dataset = create_name_for_hcp_dataset(num_nodes=p['num_nodes'],
                                                   target_var=p['target_var'],
                                                   threshold=p['threshold'],
                                                   normalisation=Normalisation(p['normalisation']),
                                                   connectivity_type=ConnType(p['connectivity_type']))
        print("Going for:", name_dataset)
        _ = HCPDataset(root=name_dataset,
                       target_var=p['target_var'],
                       num_nodes=p['num_nodes'],
                       threshold=p['threshold'],
                       normalisation=Normalisation(p['normalisation']),
                       connectivity_type=ConnType(p['connectivity_type']))
