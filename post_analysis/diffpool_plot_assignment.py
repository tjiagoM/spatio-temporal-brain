import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import fcluster

from utils_datasets import STRUCT_COLUMNS


# python diffpool_plot_assignment.py --sweep_name 100_n_diffpool
# python diffpool_plot_assignment.py --sweep_name 100_n_e_diffpool

def plot_and_save_interp(arr, name, sweep_name, case_name):
    s_df = pd.DataFrame(arr, index=STRUCT_COLUMNS, columns=STRUCT_COLUMNS)

    # First create a dummy clustermap to know how the dendrogram is created and find the right mask next
    g_obj = sns.clustermap(s_df, yticklabels=1, xticklabels=1, dendrogram_ratio=(0.1, 0.2),
                           cbar_pos=(0, 0.85, 0.02, 0.15), cmap="viridis")
    mask_array = np.full(arr.shape, False)
    mask_array[np.tril_indices(mask_array.shape[0])] = True

    mask_after = mask_array[np.argsort(g_obj.dendrogram_row.reordered_ind), :]
    mask_after = mask_after[:, np.argsort(g_obj.dendrogram_col.reordered_ind)]

    g_obj = sns.clustermap(s_df, yticklabels=1, xticklabels=1, dendrogram_ratio=(0.1, 0.2),
                           cbar_pos = (0, 0.85, 0.02,0.15), cmap = "viridis", mask = mask_after,
                           linewidths = 0.5, linecolor = (0.7, 0.7, 0.7, 0.2))

    g_obj.ax_heatmap.set_xticklabels(g_obj.ax_heatmap.get_xmajorticklabels(), fontsize=7)
    g_obj.ax_heatmap.set_yticklabels(g_obj.ax_heatmap.get_ymajorticklabels(), fontsize=7)

    if case_name == 'kmeans_clust':
        granularities = [4]
    elif sweep_name == '100_n_e_diffpool':
        granularities = [4, 8, 12]
    elif sweep_name == '100_n_diffpool':
        granularities = [4, 8, 12]

    for granularity_id in granularities:
        ord_ind = fcluster(g_obj.dendrogram_col.linkage, granularity_id, criterion='maxclust')

        tmp_df = pd.DataFrame(ord_ind, index=s_df.index, columns=['cluster'])

        for hemi_char in ['l_', 'r_']:
            t2_df = tmp_df[tmp_df.index.str.startswith(hemi_char)]
            t2_df.index = t2_df.index.map(lambda x: x.replace(hemi_char, ''))
            if case_name == 'kmeans_clust':
                t2_df.to_csv(f'results/kmeans_clust_{granularity_id}_{hemi_char}{name}.csv', index_label='label')
            else:
                t2_df.to_csv(f'results/dp_clust_{granularity_id}_{sweep_name}_{hemi_char}{name}.csv',
                             index_label='label')


    g_obj.savefig(f'figures/{case_name}_{sweep_name}_{name}.pdf')
    plt.close()


# num_males_test1 = 3305
# num_females_test1 = 3727

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_name')
    parser.add_argument('--case_name', choices=['dp_interp', 'kmeans_clust'])
    args = parser.parse_args()

    print(args)

    sweep_name = args.sweep_name
    case_name = args.case_name

    if case_name == 'dp_interp':
        s_male = np.load(f'results/dp_interp_{sweep_name}_male.npy')
        s_female = np.load(f'results/dp_interp_{sweep_name}_female.npy')
    else:
        s_male = np.load(f'results/kmeans_clust_male.npy')
        s_female = np.load(f'results/kmeans_clust_female.npy')

    s_total = s_male + s_female

    for s_arr, s_name in [(s_male, 'male'), (s_female, 'female'), (s_total, 'total')]:
        plot_and_save_interp(s_arr + s_arr.T, s_name, sweep_name, case_name)
