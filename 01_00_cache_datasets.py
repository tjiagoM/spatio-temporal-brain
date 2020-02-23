from sklearn.model_selection import ParameterGrid

from datasets import BrainDataset
from utils import create_name_for_brain_dataset, Normalisation, ConnType

if __name__ == '__main__':
    combinations = [{'num_nodes': [376],
                     'target_var': ['gender'],
                     'time_length': [490],
                     'threshold': [5, 20],
                     'connectivity_type': ['fmri'],
                     'normalisation': ['roi_norm']}]
    grid = ParameterGrid(combinations)

    for p in grid:
        name_dataset = create_name_for_brain_dataset(num_nodes=p['num_nodes'],
                                                     target_var=p['target_var'],
                                                     threshold=p['threshold'],
                                                     time_length=p['time_length'],
                                                     normalisation=Normalisation(p['normalisation']),
                                                     connectivity_type=ConnType(p['connectivity_type']))
        print("Going for:", name_dataset)
        _ = BrainDataset(root=name_dataset,
                         target_var=p['target_var'],
                         num_nodes=p['num_nodes'],
                         threshold=p['threshold'],
                         normalisation=Normalisation(p['normalisation']),
                         connectivity_type=ConnType(p['connectivity_type']))
