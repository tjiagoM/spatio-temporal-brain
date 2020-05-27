import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils_datasets import STRUCT_COLUMNS


def plot_and_save_interp(arr, name, sweep_name):
    s_df = pd.DataFrame(arr, index=STRUCT_COLUMNS, columns=STRUCT_COLUMNS)
    g_obj = sns.clustermap(s_df, yticklabels=1, xticklabels=1)
    g_obj.ax_heatmap.set_xticklabels(g_obj.ax_heatmap.get_xmajorticklabels(), fontsize=7)
    g_obj.ax_heatmap.set_yticklabels(g_obj.ax_heatmap.get_ymajorticklabels(), fontsize=7)
    g_obj.savefig(f'figures/dp_interp_{sweep_name}_{name}.pdf')
    plt.close()


num_males_test1 = 3305
num_females_test1 = 3727

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', help='run id')
parser.add_argument('--sweep_name')
args = parser.parse_args()

run_id = args.run_id
sweep_name = args.sweep_name

# run_id = 'fdy5th0d'
# sweep_name = 'no'

s_male = np.load(f'results/dp_interp_{run_id}_male.npy')
s_female = np.load(f'results/dp_interp_{run_id}_female.npy')

s_total = s_male + s_female

for s_arr, s_name in [(s_male, 'male'), (s_female, 'female'), (s_total, 'total')]:
    plot_and_save_interp(s_arr + s_arr.T, s_name, sweep_name)

s_female /= num_females_test1
s_male /= num_males_test1

s_difference = s_female - s_male

plot_and_save_interp(s_difference + s_difference.T, 'difference', sweep_name)
