import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import fcluster

from utils_datasets import STRUCT_COLUMNS

# python diffpool_plot_assignment.py --sweep_name 100_n_diffpool
# python diffpool_plot_assignment.py --sweep_name 100_n_e_diffpool

def plot_and_save_interp(arr, name, sweep_name):
    s_df = pd.DataFrame(arr, index=STRUCT_COLUMNS, columns=STRUCT_COLUMNS)
    g_obj = sns.clustermap(s_df, yticklabels=1, xticklabels=1)
    g_obj.ax_heatmap.set_xticklabels(g_obj.ax_heatmap.get_xmajorticklabels(), fontsize=7)
    g_obj.ax_heatmap.set_yticklabels(g_obj.ax_heatmap.get_ymajorticklabels(), fontsize=7)

    if sweep_name != 'nodeedge':
        ord_ind = fcluster(g_obj.dendrogram_col.linkage, 4, criterion='maxclust')

        tmp_df = pd.DataFrame(ord_ind, index=s_df.index, columns=['Cl'])
        tmp_df.index = tmp_df.index.map(lambda x: x.replace('l_', 'lh_').replace('r_', 'rh_'))
        tmp_df.loc['rh_medialwall', 'Cl'] = 0
        tmp_df.loc['lh_medialwall', 'Cl'] = 0
        tmp_df.to_csv(f'results/dp_clust_{sweep_name}_{name}.csv', index_label='label')

    g_obj.savefig(f'figures/dp_interp_{sweep_name}_{name}.pdf')
    plt.close()


#num_males_test1 = 3305
#num_females_test1 = 3727

parser = argparse.ArgumentParser()
parser.add_argument('--sweep_name')
args = parser.parse_args()

sweep_name = args.sweep_name

s_male = np.load(f'results/dp_interp_{sweep_name}_male.npy')
s_female = np.load(f'results/dp_interp_{sweep_name}_female.npy')

s_total = s_male + s_female

for s_arr, s_name in [(s_male, 'male'), (s_female, 'female'), (s_total, 'total')]:
    plot_and_save_interp(s_arr + s_arr.T, s_name, sweep_name)

#s_female /= num_females_test1
#s_male /= num_males_test1

#s_difference = s_female - s_male

#plot_and_save_interp(s_difference + s_difference.T, 'difference', sweep_name)
