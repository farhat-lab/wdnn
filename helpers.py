import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from keras.layers.convolutional import *
import keras.backend as K
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Definitions
cv_splits = 5
num_drugs = 11
mask_value = -1
num_snp = 6342
num_samples = 3619
drugs = ['rif', 'inh', 'pza', 'emb', 'str', 'cip', 'cap', 'amk', 'moxi', 'oflx', 'kan']

# Masked binary cross-entropy for single task MLP
def masked_single_bce(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    return K.mean(K.binary_crossentropy((y_pred * mask), (y_true * mask)))

# Masked accuracy as metric for both multitask and single task MLPs
def masked_accuracy(y_true, y_pred):
    total = K.sum(K.cast(K.not_equal(y_true, mask_value), K.floatx()))
    correct = K.sum(K.cast(K.equal(y_true, K.round(y_pred)), K.floatx()))
    return correct / total

# Weighted loss function for multitask model with mask
def masked_multi_weighted_bce(alpha, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    y_true_ = K.cast(K.greater(alpha, 0.), K.floatx())
    mask = K.cast(K.not_equal(alpha, 0.), K.floatx())
    num_not_missing = K.sum(mask, axis=-1)
    alpha = K.abs(alpha)
    bce = - alpha * y_true_ * K.log(y_pred) - (1.0 - alpha) * (1.0 - y_true_) * K.log(1.0 - y_pred)
    masked_bce = bce * mask
    return K.sum(masked_bce, axis=-1) / num_not_missing

# Accuracy measurement for weighted loss function with mask
def masked_weighted_accuracy(alpha, y_pred):
    total = K.sum(K.cast(K.not_equal(alpha, 0.), K.floatx()))
    y_true_ = K.cast(K.greater(alpha, 0.), K.floatx())
    mask = K.cast(K.not_equal(alpha, 0.), K.floatx())
    correct = K.sum(K.cast(K.equal(y_true_, K.round(y_pred)), K.floatx()) * mask)
    return correct / total

# Get thresholds
def get_threshold_val(y_true, y_pred, type):
    num_samples = y_pred.shape[0]
    fpr_ = []
    tpr_ = []
    thresholds = np.linspace(0,1,101)
    num_sensitive = np.sum(y_true)
    num_resistant = num_samples - num_sensitive
    for threshold in thresholds:
        fp_ = 0
        tp_ = 0
        for i in range(num_samples):
            if (y_pred[i] < threshold):
                if (y_true[i] == 1): fp_ += 1
                if (y_true[i] == 0): tp_ += 1
        fpr_.append(fp_ / float(num_sensitive))
        tpr_.append(tp_ / float(num_resistant))
    fpr_ = np.array(fpr_)
    tpr_ = np.array(tpr_)
    if type == "spec_90":
        valid_inds = np.where(fpr_ <= 0.1)
        sens_spec_sum = (1 - fpr_) + tpr_
        best_sens_spec_sum = np.max(sens_spec_sum[valid_inds])
        best_inds = np.where(best_sens_spec_sum == sens_spec_sum[valid_inds])
        if best_inds[0].shape[0] == 1:
            best_sens_spec_ind = best_inds
        else:
            best_sens_spec_ind = np.array(np.squeeze(best_inds))[-1]
        return {'threshold':thresholds[valid_inds][best_sens_spec_ind],
                'spec':1 - fpr_[valid_inds][best_sens_spec_ind],
                'sens':tpr_[valid_inds][best_sens_spec_ind]}
    if type == "max":
        sens_spec_sum = (1 - fpr_) + tpr_
        best_ind = np.argmax(sens_spec_sum)
        return {'threshold':thresholds[best_ind],
                'spec':1 - fpr_[best_ind],
                'sens':tpr_[best_ind]}

# Used to get "retroactive" threshold from each fold's validation set to (average and)
# apply to independent test set.
def get_threshold_val(y_true, y_pred):
    num_samples = y_pred.shape[0]
    fpr_ = []
    tpr_ = []
    thresholds = np.linspace(0,1,101)
    num_sensitive = np.sum(y_true)
    num_resistant = num_samples - num_sensitive
    for threshold in thresholds:
        fp_ = 0
        tp_ = 0
        for i in range(num_samples):
            if (y_pred[i] < threshold):
                if (y_true[i] == 1): fp_ += 1
                if (y_true[i] == 0): tp_ += 1
        fpr_.append(fp_ / float(num_sensitive))
        tpr_.append(tp_ / float(num_resistant))
    fpr_ = np.array(fpr_)
    tpr_ = np.array(tpr_)
    valid_inds = np.where(fpr_ <= 0.1)
    sens_spec_sum = (1 - fpr_) + tpr_
    best_sens_spec_sum = np.max(sens_spec_sum[valid_inds])
    best_inds = np.where(best_sens_spec_sum == sens_spec_sum[valid_inds])
    if best_inds[0].shape[0] == 1:
        best_sens_spec_ind = best_inds
    else:
        best_sens_spec_ind = np.array(np.squeeze(best_inds))[-1]
    return {'threshold':thresholds[valid_inds][best_sens_spec_ind],
            'spec':1 - fpr_[valid_inds][best_sens_spec_ind],
            'sens':tpr_[valid_inds][best_sens_spec_ind]}

# sensitivity >=  0.90; 1 - sensitivty = fpr; fpr <= 0.10
# Threshold picking based on training data and applies to the validation data.
# Used in cross-validation to determine threshold for validation set in each fold.
def get_threshold(y_true_train, y_pred_train, y_true_test, y_pred_test):
    num_samples_train = y_pred_train.shape[0]
    num_samples_test = y_pred_test.shape[0]
    fpr_ = []
    tpr_ = []
    thresholds = np.linspace(0,1,101)
    num_sensitive = np.sum(y_true_train)
    num_resistant = num_samples_train - num_sensitive
    for threshold in thresholds:
        fp_ = 0
        tp_ = 0
        for i in range(num_samples_train):
            if (y_pred_train[i] < threshold):
                if (y_true_train[i] == 1): fp_ += 1
                if (y_true_train[i] == 0): tp_ += 1
        fpr_.append(fp_ / float(num_sensitive))
        tpr_.append(tp_ / float(num_resistant))
    fpr_ = np.array(fpr_)
    tpr_ = np.array(tpr_)
    valid_inds = np.where(fpr_ <= 0.1)
    sens_spec_sum = (1 - fpr_) + tpr_
    best_sens_spec_sum = np.max(sens_spec_sum[valid_inds])
    best_inds = np.where(best_sens_spec_sum == sens_spec_sum[valid_inds])
    if best_inds[0].shape[0] == 1:
        best_sens_spec_ind = best_inds
    else:
        best_sens_spec_ind = np.array(np.squeeze(best_inds))[-1]
    fp_final = 0
    tp_final = 0
    final_threshold = thresholds[valid_inds][best_sens_spec_ind]
    num_sensitive_final = np.sum(y_true_test)
    num_resistant_final = num_samples_test - num_sensitive_final
    for i in range(num_samples_test):
        if (y_pred_test[i] < final_threshold):
            if (y_true_test[i] == 1): fp_final += 1
            if (y_true_test[i] == 0): tp_final += 1
    fp_final /= float(num_sensitive_final)
    tp_final /= float(num_resistant_final)
    return {'threshold':final_threshold,
            'spec':1 - fp_final,
            'sens':tp_final}

# Calculates the sensitivity/specificity based on labels, predictions, and inputed threshold.
def get_sens_spec_from_threshold(y_true_test, y_pred_test, final_threshold):
    num_samples_test = y_pred_test.shape[0]
    fp_final = 0
    tp_final = 0
    num_sensitive_final = np.sum(y_true_test)
    num_resistant_final = num_samples_test - num_sensitive_final
    for i in range(num_samples_test):
        if (y_pred_test[i] < final_threshold):
            if (y_true_test[i] == 1): fp_final += 1
            if (y_true_test[i] == 0): tp_final += 1
    fp_final /= float(num_sensitive_final)
    tp_final /= float(num_resistant_final)
    return {'threshold':final_threshold,
            'spec':1 - fp_final,
            'sens':tp_final}

# Function to return the full list of TPRs and FPRs given labels and predictions.
def plot_roc_auc(drug, y_true, y_pred):
    num_samples = y_pred.shape[0]
    fpr_ = []
    tpr_ = []
    thresholds = np.linspace(0,1,101)
    num_sensitive = np.sum(y_true)
    num_resistant = num_samples - num_sensitive
    for threshold in thresholds:
        fp_ = 0
        tp_ = 0
        for i in range(num_samples):
            if (y_pred[i] < threshold):
                if (y_true[i] == 1): fp_ += 1
                if (y_true[i] == 0): tp_ += 1
        fpr_.append(fp_ / float(num_sensitive))
        tpr_.append(tp_ / float(num_resistant))
    return {'tpr_list':tpr_, 'fpr_list':fpr_}

# Function to plot ROC curve based on TPRs and FPRs.
def final_plot_roc_auc(drugs, fpr_list, tpr_list):
    for i, drug in enumerate(drugs):
        fig = plt.figure()
        plt.plot(fpr_list[:,i], tpr_list[:,i], label='Multi WDNN')
        plt.plot(fpr_list[:, i + 10], tpr_list[:, i + 10], label='RF')
        plt.plot(fpr_list[:, i + 20], tpr_list[:, i + 20], label='LR')
        plt.plot(fpr_list[:, i + 30], tpr_list[:, i + 30], label='Single WDNN')
        plt.plot(fpr_list[:, i + 40], tpr_list[:, i + 40], label='Preselected MLP')
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'black')
        plt.xlim([-.02, 1.02])
        plt.ylim([-.02, 1.02])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        fig.savefig(str(drug)+'.png')
    return None

# Gene names
genes = ['ahpC', 'alr', 'ddl', 'embA', 'embB', 'embC', 'ethA', 'gid', 'gyrA', 'gyrB', 'inhA', 'iniA',
         'iniB', 'iniC', 'kasA', 'katG', 'murA-rrs', 'fabG1', 'ndh', 'pncA', 'rpoB', 'rpsL', 'rrl',
         'rrs', 'thyA', 'tlyA', 'gyrB-gyrA', 'gyrA-Rv0007', 'iniB-iniA-iniC', 'iniB-iniA', 'iniC-lpqJ',
         'rpoB-rpoC', 'fabG1-inhA', 'rrs-rrl', 'rrl-rrf', 'inhA-hemZ', 'katG-furA', 'kasA-kasB',
         'ahpC-ahpD', 'dfrA-thyA', 'alr-Rv3792', 'embA-embB', 'embB-Rv3796', 'menG-ethA', 'rpsA', 'eis',
         "oxyR'", 'acpM']

# Get gene associated with particular mutation
def get_gene(x):
    splitted = x.split("_")
    if list(set(splitted).intersection(genes)) == []:
        if 'ndhA' in splitted or 'mfd' in splitted or 'whiB6':
            return None
        #return None
        raise Exception(splitted)
    type = '-'.join(list(set(splitted).intersection(genes)))
    return '-'.join(sorted(type.split('-')))

# Get gene dictionary
def get_gene_dict(derived_names):
    gene_dict = {}
    for snp in derived_names:
        if not get_gene(snp):
            continue
        gene_dict.setdefault(get_gene(snp), []).append(snp)
    return gene_dict

# Get dictionary by mutation location and type of mutation
def get_final_dict(gene_dict):
    final_dict = {}
    for gene, muts in gene_dict.iteritems():
        for mut in muts:
            split = mut.split("_")
            if 'P' in split or 'I' in split:
                if 'DEL' in split or 'INS' in split:
                    final_dict.setdefault(gene + '_NC_indel', []).append(mut)
                    # is noncoding indel
                else:
                    final_dict.setdefault(gene + '_NC_snp', []).append(mut)
                    # is noncoding snp
            elif 'F' in split or 'CF' in split:
                final_dict.setdefault(gene + '_F_indel', []).append(mut)
                # is coding frameshift
            elif 'CI' in split or 'N' in split or 'NF' in split:
                final_dict.setdefault(gene + '_NF_indel', []).append(mut)
                # is coding not frameshift
            else:
                final_dict.setdefault(gene + '_C_snp', []).append(mut)
                # is coding snp
    return final_dict

def ensemble(X, y, function):
    preds = np.zeros_like(y, dtype=np.float)
    for i in range(100):
        preds += np.squeeze(np.array(function([X, 1])), axis=0)
    return preds / 100.

# Miscellaneous helper functions
def make_data_array():
    return np.zeros((num_drugs, cv_splits), dtype=np.float)

def get_mean(data_array):
    return np.mean(data_array, axis=1)

def get_stderr(data_array):
    return np.std(data_array, axis=1) / np.sqrt(cv_splits)

def remove_cip(data_array):
    return data_array[data_array != 0]


