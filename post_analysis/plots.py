import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import friedmanchisquare

from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report

NAME_MODEL_LOSS = 'gender_2_0_loss_D_0__A_relu__P_mean__CS_entire__CHC_8__FS_True__GCN_False__GAT_False_0.0001_0.005_20_5_roi_norm_150_False_272_struct'
NAME_MODEL_AUC = 'gender_2_0_auc_D_0__A_relu__P_mean__CS_entire__CHC_8__FS_True__GCN_False__GAT_False_0.0001_0.5_20_5_roi_norm_150_False_272_struct'

labels_auc = np.load('labels_' + NAME_MODEL_AUC + '.npy')
preds_auc = np.load('predictions_' + NAME_MODEL_AUC + '.npy')

labels_loss = np.load('labels_' + NAME_MODEL_LOSS + '.npy')
preds_loss = np.load('predictions_' + NAME_MODEL_LOSS + '.npy')

fpr_auc, tpr_auc, _ = roc_curve(labels_auc, preds_auc)
roc_auc = auc(fpr_auc, tpr_auc)
fpr_loss, tpr_loss, _ = roc_curve(labels_loss, preds_loss)
roc_loss = auc(fpr_loss, tpr_loss)

plt.figure()
plt.plot(fpr_auc, tpr_auc, color='darkorange', lw=2, label=f'ROC curve for AUC (area = {round(roc_auc, 3)})')
plt.plot(fpr_loss, tpr_loss, color='darkred', lw=2, label=f'ROC curve for Loss (area = {round(roc_loss, 3)})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


#######################
#######################

ordered_named_results = [
'gender_1_0_loss_D_0.5__A_relu__P_mean__CS_entire__CHC_8__FS_True__GCN_False__GAT_False_1e-05_0.005_30_5_roi_norm_500_False_50_fmri.npy',
'gender_1_0_loss_D_0.5__A_relu__P_mean__CS_entire__CHC_8__FS_True__GCN_False__GAT_False_1e-05_0.005_30_20_roi_norm_500_False_50_fmri.npy',
'gender_1_0_loss_D_0.7__A_relu__P_diff_pool__CS_entire__CHC_8__FS_True__GCN_False__GAT_False_0.0001_0.005_30_5_roi_norm_500_False_50_fmri.npy',
'gender_1_0_loss_D_0.7__A_relu__P_diff_pool__CS_entire__CHC_8__FS_True__GCN_False__GAT_False_0.0001_0_30_20_roi_norm_500_False_50_fmri.npy',

'gender_1_0_loss_D_0__A_relu__P_mean__CS_entire__CHC_8__FS_True__GCN_True__GAT_False_1e-05_0_30_5_roi_norm_500_False_50_fmri.npy',
'gender_1_0_loss_D_0__A_relu__P_mean__CS_entire__CHC_8__FS_True__GCN_True__GAT_False_1e-05_0_30_20_roi_norm_500_False_50_fmri.npy',
'gender_1_0_loss_D_0.5__A_relu__P_diff_pool__CS_entire__CHC_8__FS_True__GCN_True__GAT_False_0.0001_0_30_5_roi_norm_500_False_50_fmri.npy',
'gender_1_0_loss_D_0.5__A_relu__P_diff_pool__CS_entire__CHC_8__FS_True__GCN_True__GAT_False_0.0001_0.005_30_20_roi_norm_500_False_50_fmri.npy',


'gender_2_0_loss_D_0.5__A_relu__P_mean__CS_entire__CHC_8__FS_True__GCN_False__GAT_False_1e-05_0_30_5_roi_norm_500_False_50_fmri.npy',
'gender_2_0_loss_D_0.5__A_relu__P_mean__CS_entire__CHC_8__FS_True__GCN_False__GAT_False_1e-05_0_30_20_roi_norm_500_False_50_fmri.npy',
'gender_2_0_loss_D_0.5__A_relu__P_diff_pool__CS_entire__CHC_8__FS_True__GCN_False__GAT_False_0.0001_0.005_30_5_roi_norm_500_False_50_fmri.npy',
'gender_2_0_loss_D_0.7__A_relu__P_diff_pool__CS_entire__CHC_8__FS_True__GCN_False__GAT_False_0.0001_0.005_30_20_roi_norm_500_False_50_fmri.npy',

'gender_2_0_loss_D_0__A_relu__P_mean__CS_entire__CHC_8__FS_True__GCN_True__GAT_False_1e-05_0_30_5_roi_norm_500_False_50_fmri.npy',
'gender_2_0_loss_D_0__A_relu__P_mean__CS_entire__CHC_8__FS_True__GCN_True__GAT_False_1e-05_0_30_20_roi_norm_500_False_50_fmri.npy',
'gender_2_0_loss_D_0.5__A_relu__P_diff_pool__CS_entire__CHC_8__FS_True__GCN_True__GAT_False_0.0001_0.005_30_5_roi_norm_500_False_50_fmri.npy',
'gender_2_0_loss_D_0.5__A_relu__P_diff_pool__CS_entire__CHC_8__FS_True__GCN_True__GAT_False_0.0001_0_30_20_roi_norm_500_False_50_fmri.npy',


'gender_3_0_loss_D_0__A_relu__P_mean__CS_entire__CHC_8__FS_True__GCN_False__GAT_False_1e-05_0.005_30_5_roi_norm_500_False_50_fmri.npy',
'gender_3_0_loss_D_0__A_relu__P_mean__CS_entire__CHC_8__FS_True__GCN_False__GAT_False_1e-05_0.005_30_20_roi_norm_500_False_50_fmri.npy',
'gender_3_0_loss_D_0.7__A_relu__P_diff_pool__CS_entire__CHC_8__FS_True__GCN_False__GAT_False_0.0001_0_30_5_roi_norm_500_False_50_fmri.npy',
'gender_3_0_loss_D_0.7__A_relu__P_diff_pool__CS_entire__CHC_8__FS_True__GCN_False__GAT_False_0.0001_0.005_30_20_roi_norm_500_False_50_fmri.npy',

'gender_3_0_loss_D_0.7__A_relu__P_mean__CS_entire__CHC_8__FS_True__GCN_True__GAT_False_0.0001_0.005_30_5_roi_norm_500_False_50_fmri.npy',
'gender_3_0_loss_D_0__A_relu__P_mean__CS_entire__CHC_8__FS_True__GCN_True__GAT_False_1e-05_0_30_20_roi_norm_500_False_50_fmri.npy',
'gender_3_0_loss_D_0.5__A_relu__P_diff_pool__CS_entire__CHC_8__FS_True__GCN_True__GAT_False_0.0001_0_30_5_roi_norm_500_False_50_fmri.npy',
'gender_3_0_loss_D_0.7__A_relu__P_diff_pool__CS_entire__CHC_8__FS_True__GCN_True__GAT_False_0.0001_0_30_20_roi_norm_500_False_50_fmri.npy',


'gender_4_0_loss_D_0__A_relu__P_mean__CS_entire__CHC_8__FS_True__GCN_False__GAT_False_0.0001_0.005_30_5_roi_norm_500_False_50_fmri.npy',
'gender_4_0_loss_D_0__A_relu__P_mean__CS_entire__CHC_8__FS_True__GCN_False__GAT_False_0.0001_0.005_30_20_roi_norm_500_False_50_fmri.npy',
'gender_4_0_loss_D_0.5__A_relu__P_diff_pool__CS_entire__CHC_8__FS_True__GCN_False__GAT_False_0.0001_0_30_5_roi_norm_500_False_50_fmri.npy',
'gender_4_0_loss_D_0.5__A_relu__P_diff_pool__CS_entire__CHC_8__FS_True__GCN_False__GAT_False_0.0001_0.005_30_20_roi_norm_500_False_50_fmri.npy',

'gender_4_0_loss_D_0.7__A_relu__P_mean__CS_entire__CHC_8__FS_True__GCN_True__GAT_False_0.0001_0_30_5_roi_norm_500_False_50_fmri.npy',
'gender_4_0_loss_D_0.7__A_relu__P_mean__CS_entire__CHC_8__FS_True__GCN_True__GAT_False_0.0001_0_30_20_roi_norm_500_False_50_fmri.npy',
'gender_4_0_loss_D_0__A_relu__P_diff_pool__CS_entire__CHC_8__FS_True__GCN_True__GAT_False_0.0001_0.005_30_5_roi_norm_500_False_50_fmri.npy',
'gender_4_0_loss_D_0.5__A_relu__P_diff_pool__CS_entire__CHC_8__FS_True__GCN_True__GAT_False_0.0001_0.005_30_20_roi_norm_500_False_50_fmri.npy',


'gender_5_0_loss_D_0.7__A_relu__P_mean__CS_entire__CHC_8__FS_True__GCN_False__GAT_False_0.0001_0_30_5_roi_norm_500_False_50_fmri.npy',
'gender_5_0_loss_D_0.7__A_relu__P_mean__CS_entire__CHC_8__FS_True__GCN_False__GAT_False_0.0001_0_30_20_roi_norm_500_False_50_fmri.npy',
'gender_5_0_loss_D_0.7__A_relu__P_diff_pool__CS_entire__CHC_8__FS_True__GCN_False__GAT_False_0.0001_0.005_30_5_roi_norm_500_False_50_fmri.npy',
'gender_5_0_loss_D_0.5__A_relu__P_diff_pool__CS_entire__CHC_8__FS_True__GCN_False__GAT_False_0.0001_0_30_20_roi_norm_500_False_50_fmri.npy',

'gender_5_0_loss_D_0.7__A_relu__P_mean__CS_entire__CHC_8__FS_True__GCN_True__GAT_False_0.0001_0_30_5_roi_norm_500_False_50_fmri.npy',
'gender_5_0_loss_D_0.7__A_relu__P_mean__CS_entire__CHC_8__FS_True__GCN_True__GAT_False_0.0001_0_30_20_roi_norm_500_False_50_fmri.npy',
'gender_5_0_loss_D_0.7__A_relu__P_diff_pool__CS_entire__CHC_8__FS_True__GCN_True__GAT_False_0.0001_0.005_30_5_roi_norm_500_False_50_fmri.npy',
'gender_5_0_loss_D_0.7__A_relu__P_diff_pool__CS_entire__CHC_8__FS_True__GCN_True__GAT_False_0.0001_0.005_30_20_roi_norm_500_False_50_fmri.npy'
]

import warnings
warnings.filterwarnings("ignore")
for i, name in enumerate(['mean_5', 'mean_20', 'diff_5', 'diff_20', 'gcn_mean_5', 'gcn_mean_20', 'gcn_diff_5', 'gcn_diff_20']):
    print(name, ':')

    all_rocs = []
    all_specificities = []
    all_sensitivities = []
    for fold, ord_index in enumerate(range(i, 40, 8)):
        name = ordered_named_results[ord_index]
        labels = np.load('results/labels_' + name)
        predictions = np.load('results/predictions_' + name)
        pred_binary = np.where(predictions > 0.5, 1, 0)

        roc_auc = round(roc_auc_score(labels, predictions), 4)
        all_rocs.append(roc_auc)

        report = classification_report(labels, pred_binary, output_dict=True)
        sens = round(report['1.0']['recall'], 4)
        all_sensitivities.append(sens)
        spec = round(report['0.0']['recall'], 4)
        all_specificities.append(spec)

        print('fold', fold, ':', roc_auc, 'Sens:', sens, 'Spec:', spec)

    print("roc :", round(np.mean(all_rocs), 3), "(", round(np.std(all_rocs), 3), ")")
    print("sens :", round(np.mean(all_sensitivities), 3), "(", round(np.std(all_sensitivities), 3), ")")
    print("spec :", round(np.mean(all_specificities), 3), "(", round(np.std(all_specificities), 3), ")")
    print()


########################
########################
# Adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
from scipy import interp
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
colours_plot = ['r', 'g', 'c', 'y', 'm']

fig, ax = plt.subplots()
for fold, ord_index in enumerate(range(4, 40, 8)):
    name = ordered_named_results[ord_index]
    labels = np.load('results/labels_' + name)
    predictions = np.load('results/predictions_' + name)

    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_val = auc(fpr, tpr)

    ax.plot(fpr, tpr, lw=2, alpha=0.5, label=f'Roc Fold {fold+1}', color=colours_plot[fold])
    interp_tpr = interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(roc_val)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])#,
       #title="Receiver Operating Characteristic curves for ST-mean")
ax.legend(loc="lower right")
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
plt.tight_layout()
plt.savefig('rocs_st_mean.pdf', bbox_inches = 'tight', pad_inches = 0)
plt.show()

########################
########################
def get_adj_50_path(person, index):
    return f'../../../space/hcp_50_timeseries/processed_4_split_50/{person}_{index}.npy'

def get_50_ts_path(person):
    return f'../hcp_timecourses/3T_HCP1200_MSMAll_d50_ts2/{person}.txt'
person = 100206
all_ts = np.genfromtxt(get_50_ts_path(person))
t1 = all_ts[:1200, :]

import pandas as pd
plt.rcParams.update({'font.size': 17})
#fig, axes = plt.subplots(1, 2)
#
plt.figure(figsize=(6,6))
data = {'ICA_1' : t1[:,0], 'ICA_2' : t1[:,1], 'ICA_25' : t1[:,24], 'ICA_50' : t1[:,49]}
df = pd.DataFrame(data)
axes = df.plot(subplots=True, figsize=(6, 6), legend=False)
for ax in axes:
    ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig('50_50_timeseries_example.pdf', bbox_inches = 'tight', pad_inches = 0)
#
corr_arr = np.load(get_adj_50_path(100206, 0))
corr_arr[np.tril_indices(50)] = 0
corr_arr_5 = corr_arr.copy()

num_to_filter_5 = int((5 / 100.0) * (50 * (50 - 1) / 2))
num_to_filter_20 = int((20 / 100.0) * (50 * (50 - 1) / 2))
indices = np.where(corr_arr)
sorted_indices = np.argsort(corr_arr[indices])[::-1]
corr_arr[(indices[0][sorted_indices][num_to_filter_20:], indices[1][sorted_indices][num_to_filter_20:])] = 0
corr_arr[corr_arr > 0] = 1

corr_arr_5[(indices[0][sorted_indices][num_to_filter_5:], indices[1][sorted_indices][num_to_filter_5:])] = 0
corr_arr_5[corr_arr_5 > 0] = 2

corr_arr[corr_arr_5 == 2] = 2

import matplotlib.patches as mpatches
plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(6,6))
im = plt.imshow(corr_arr, cmap='viridis')

patches = [ mpatches.Patch(color=im.cmap(im.norm(0)), label='No correlation'),
            mpatches.Patch(color=im.cmap(im.norm(2)), label='With 5% threshold'),
            mpatches.Patch(color=im.cmap(im.norm(1)), label='With 20% threshold')
            ]
# put those patched as legend-handles into the legend
plt.legend(handles=patches, loc='lower left', prop={"size":20})
plt.tight_layout()
plt.savefig('50_threshold_example.pdf', bbox_inches = 'tight', pad_inches = 0)


#################
######
#All AUCs
mean_cnn = [0.6895, 0.6388, 0.6871, 0.7488, 0.6807]
mean_cnn_gcn5 = [0.7206, 0.6477, 0.6855, 0.7581, 0.6935]
mean_cnn_gcn20 = [0.7216, 0.6460, 0.6840, 0.7588, 0.6897]
xgboost_4split = [0.7875, 0.7723, 0.7853, 0.7859, 0.7950]

stat, p = friedmanchisquare(mean_cnn,
                            mean_cnn_gcn5,
                            mean_cnn_gcn20)
print(stat, p)


####################################################################
##########################################################################
# Check graphs and timeseries
from utils import Normalisation, ConnType, AnalysisType, EncodingStrategy, DatasetType
import networkx as nx
import os
import matplotlib.pyplot as plt
import numpy as np
from main_loop import generate_dataset
from datasets import UKBDataset
import torch
import pandas as pd

def to_networkx2(data, node_attrs=None, edge_attrs=None, to_undirected=False,
                remove_self_loops=False):
    r"""Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`networkx.DiGraph` if :attr:`to_undirected` is set to :obj:`True`, or
    an undirected :obj:`networkx.Graph` otherwise.

    Args:
        data (torch_geometric.data.Data): The data object.
        node_attrs (iterable of str, optional): The node attributes to be
            copied. (default: :obj:`None`)
        edge_attrs (iterable of str, optional): The edge attributes to be
            copied. (default: :obj:`None`)
        to_undirected (bool, optional): If set to :obj:`True`, will return a
            a :obj:`networkx.Graph` instead of a :obj:`networkx.DiGraph`. The
            undirected graph will correspond to the upper triangle of the
            corresponding adjacency matrix. (default: :obj:`False`)
        remove_self_loops (bool, optional): If set to :obj:`True`, will not
            include self loops in the resulting graph. (default: :obj:`False`)
    """

    if to_undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    G.add_nodes_from(range(data.num_nodes))

    values = {}
    for key, item in data:
        if torch.is_tensor(item):
            values[key] = item.squeeze().tolist()
        else:
            values[key] = item
        if isinstance(values[key], (list, tuple)) and len(values[key]) == 1:
            values[key] = item[0]

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):

        if to_undirected and v > u:
            continue

        if remove_self_loops and u == v:
            continue

        G.add_edge(u, v)
        for key in edge_attrs if edge_attrs is not None else []:
            G[u][v][key] = values[key][i]

    for key in node_attrs if node_attrs is not None else []:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})

    return G

# For cycle to quickly explore the graphs and to chose a few
for num_nodes, time_length in [(68, 490)]: #(50, 1200),
    run_cfg = {
        'num_nodes': num_nodes,
        'time_length': time_length,
        'target_var' : 'gender',
        'param_threshold' : 10,
        'param_normalisation' : Normalisation('subject_norm'),
        'param_conn_type' : ConnType('fmri'),
        'analysis_type' : AnalysisType('st_unimodal'),
        'param_encoding_strategy' : EncodingStrategy('none'),
        'dataset_type' : DatasetType('ukb'),
        'edge_weights' : True
    }

    dataset: UKBDataset = generate_dataset(run_cfg)

    female_ind = [ind for ind, data in enumerate(dataset) if data.y == 0]
    male_ind = [ind for ind, data in enumerate(dataset) if data.y == 1]

    np.random.seed(seed=11)

    rand_indices = np.random.randint(low=0, high=min([len(female_ind), len(male_ind)]), size=15).tolist()

    data_0 = dataset[female_ind][rand_indices]
    data_1 = dataset[male_ind][rand_indices]

    for i, rand_ind in enumerate(rand_indices):
        g_fem = to_networkx2(data_0[i], to_undirected=True, remove_self_loops=True, edge_attrs=['edge_attr'])
        g_mal = to_networkx2(data_1[i], to_undirected=True, remove_self_loops=True, edge_attrs=['edge_attr'])

        for sex_type, G in [['Female', g_fem], ['Male', g_mal]]:
            fig, ax = plt.subplots(1, 2, figsize=(20, 10))

            # Getting the values of all edges in order to scale them in range [0, 1]
            all_edges = []
            for edge in G.edges(data='edge_attr'):
                all_edges.append(edge[2])

            ax[0].set_axis_off()
            ax[0].set_title('Spring Layout, 500 iters, k=5')
            pos = nx.spring_layout(G, k=5, iterations=500, scale=5)
            nx.draw_networkx(G, pos, ax=ax[0], with_labels=False, node_size=75, edgelist=[], arrows=False)
            # width makes the weight value between [0, 1], and scale it by 2
            for edge in G.edges(data='edge_attr'):
                nx.draw_networkx_edges(G, pos, ax=ax[0], edgelist=[edge], width=2*(edge[2] - min(all_edges)) / (max(all_edges) - min(all_edges)), arrows=False)

            ax[1].set_axis_off()
            ax[1].set_title('Kamada Kawai Layout')
            pos = nx.kamada_kawai_layout(G)
            nx.draw_networkx(G, pos, ax=ax[1], with_labels=False, node_size=75, edgelist=[], arrows=False)
            for edge in G.edges(data='edge_attr'):
                nx.draw_networkx_edges(G, pos, ax=ax[1], edgelist=[edge], width=2*(edge[2] - min(all_edges)) / (max(all_edges) - min(all_edges)), arrows=False)

            fig.suptitle(f'Random Person {rand_ind}, {sex_type} with {run_cfg["param_threshold"]}% threshold', fontsize=20)
            plt.tight_layout()

            plt.savefig(os.path.join('../figures', f'graph_{num_nodes}_{sex_type}_{i}.png'))
            plt.close()

#
# Plot one specific graph and some of its timeseries
#
run_cfg = {
    'num_nodes': 68,
    'time_length': 490,
    'target_var' : 'gender',
    'param_threshold' : 10,
    'param_normalisation' : Normalisation('subject_norm'),
    'param_conn_type' : ConnType('fmri'),
    'analysis_type' : AnalysisType('st_unimodal'),
    'param_encoding_strategy' : EncodingStrategy('none'),
    'dataset_type' : DatasetType('ukb'),
    'edge_weights' : True
}
STRUCT_SIMPLE_NAMES = ['L.BSTS', 'L.CACG', 'L.CMFG', 'L.CU', 'L.EC', 'L.FG', 'L.IPG', 'L.ITG', 'L.ICG', 'L.LOG', 'L.LOFG', 'L.LG', 'L.MOFG', 'L.MTG', 'L.PHIG', 'L.PaCG', 'L.POP', 'L.POR', 'L.PTR', 'L.PCAL', 'L.PoCG', 'L.PCG', 'L.PrCG', 'L.PCU', 'L.RACG', 'L.RMFG', 'L.SFG', 'L.SPG', 'L.STG', 'L.SMG', 'L.FP', 'L.TP', 'L.TTG', 'L.IN', 'R.BSTS', 'R.CACG', 'R.CMFG', 'R.CU', 'R.EC', 'R.FG', 'R.IPG', 'R.ITG', 'R.ICG', 'R.LOG', 'R.LOFG', 'R.LG', 'R.MOFG', 'R.MTG', 'R.PHIG', 'R.PaCG', 'R.POP', 'R.POR', 'R.PTR', 'R.PCAL', 'R.PoCG', 'R.PCG', 'R.PrCG', 'R.PCU', 'R.RACG', 'R.RMFG', 'R.SFG', 'R.SPG', 'R.STG', 'R.SMG', 'R.FP', 'R.TP', 'R.TTG', 'R.IN']

dataset: UKBDataset = generate_dataset(run_cfg)
male_ind = [ind for ind, data in enumerate(dataset) if data.y == 1]
data_0 = dataset[male_ind][3775]

G = to_networkx2(data_0, to_undirected=True, remove_self_loops=True, edge_attrs=['edge_attr'])
mapping = {ind: STRUCT_SIMPLE_NAMES[ind] for ind in range(len(STRUCT_SIMPLE_NAMES))}
nx.relabel_nodes(G, mapping, copy=False)
np.save('figures/example_graph.npy', nx.to_numpy_matrix(G, weight='edge_attr'))
#pd.DataFrame(nx.to_numpy_matrix(G, weight='edge_attr'), index=STRUCT_COLUMNS, columns=STRUCT_COLUMNS).to_csv('figures/example_graph.csv')
all_edges = []
for edge in G.edges(data='edge_attr'):
    all_edges.append(edge[2])

_, ax = plt.subplots(figsize=(7, 7))
pos = nx.spring_layout(G, k=12, iterations=1000)
#pos = nx.kamada_kawai_layout(G, weight='edge_attr')
##pos = nx.nx_agraph.graphviz_layout(G)
nx.draw_networkx(G, pos, ax=ax, with_labels=True, node_size=575, font_size=10, edgelist=[], arrows=False)
# width makes the weight value between [0, 1], and scale it by 2
for edge in G.edges(data='edge_attr'):
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[edge], width=3*(edge[2] - min(all_edges)) / (max(all_edges) - min(all_edges)), arrows=False)

#fig.suptitle(f'Random Person {rand_ind}, {sex_type} with {run_cfg["param_threshold"]}% threshold', fontsize=20)
plt.axis('off')
plt.tight_layout()
#plt.savefig(os.path.join('figures', f'graph_example.pdf'), bbox_inches = 'tight', pad_inches = 0)
plt.show()
plt.close()


# Timeseries
from utils_datasets import STRUCT_COLUMNS
plt.figure(figsize=(6,6))
ts = data_0.x
data = {STRUCT_COLUMNS[5] : ts[5, :], STRUCT_COLUMNS[15] : ts[15, :], STRUCT_COLUMNS[37] : ts[37, :], STRUCT_COLUMNS[40] : ts[40, :]}
df = pd.DataFrame(data)
# To match graph in paper
specific_colours = [(0.862745098039216, 0.07843137254902, 0.07843137254902, 1),
                    (0.588235294117647, 0.588235294117647, 0.784313725490196, 1),
                    (1, 0.501960784313726, 0, 1),
                    (0.313725490196078, 0.627450980392157, 0.07843137254902, 1)
                    ]
axes = df.plot(subplots=True, figsize=(7, 7), legend=False, color=specific_colours)
for ax in axes:
    ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join('../figures', f'ts_example.pdf'), bbox_inches ='tight', pad_inches = 0)
plt.close()



###
###
### Plot TCN kernels
###
###
import torch
import wandb
import matplotlib.pyplot as plt
from main_loop import generate_st_model
from model import SpatioTemporalModel
from utils import change_w_config_, create_name_for_model


# this run_id correspond to 2nd fold of 100_n_e_diffpool
run_id = 'nxqb9kvj'
api = wandb.Api()
best_run = api.run(f'/st-team/spatio-temporal-brain/runs/{run_id}')
w_config = best_run.config
inner_fold_for_val: int = 1

change_w_config_(w_config)
w_config['device_run'] = 'cuda'

model: SpatioTemporalModel = generate_st_model(w_config, for_test=True)
model_saving_path: str = create_name_for_model(target_var=w_config['target_var'],
                                               model=model,
                                               outer_split_num=w_config['fold_num'],
                                               inner_split_num=inner_fold_for_val,
                                               n_epochs=w_config['num_epochs'],
                                               threshold=w_config['threshold'],
                                               batch_size=w_config['batch_size'],
                                               num_nodes=w_config['num_nodes'],
                                               conn_type=w_config['param_conn_type'],
                                               normalisation=w_config['param_normalisation'],
                                               analysis_type=w_config['analysis_type'],
                                               metric_evaluated='loss',
                                               dataset_type=w_config['dataset_type'],
                                               lr=w_config['param_lr'],
                                               weight_decay=w_config['param_weight_decay'],
                                               edge_weights=w_config['edge_weights'])

model.load_state_dict(torch.load(model_saving_path, map_location=w_config['device_run']))
model.eval()

import seaborn as sns

weights_to_plot = [
    model.temporal_conv.network[0].conv1.weight.squeeze(1).detach().cpu().numpy(),
    model.temporal_conv.network[0].conv2.weight.reshape(16, -1).detach().cpu().numpy()
]
weights_figsize = ((8, 7), (20, 5))
weights_names = ('conv1', 'conv2')

for kernel, figsize, name in zip(weights_to_plot, weights_figsize, weights_names):
    _, ax = plt.subplots(figsize=figsize)
    sns.heatmap(kernel,
                ax=ax,
                yticklabels=False, xticklabels=False,
                cmap='viridis')
    plt.savefig(f'figures/tcn_{name}.pdf', bbox_inches = 'tight', pad_inches = 0)
    plt.close()