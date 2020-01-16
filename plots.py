import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import roc_curve, auc

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