from helpers import *
import pandas as pd
import numpy as np
from keras.layers import Dense, Dropout, Input, merge
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers

# Definitions
crit_value = 0.05
num_permute = 100000
num_drugs = 11

# Gene names
genes = ['ahpC', 'alr', 'ddl', 'embA', 'embB', 'embC', 'ethA', 'gid', 'gyrA', 'gyrB', 'inhA', 'iniA',
         'iniB', 'iniC', 'kasA', 'katG', 'murA-rrs', 'fabG1', 'ndh', 'pncA', 'rpoB', 'rpsL', 'rrl',
         'rrs', 'thyA', 'tlyA', 'gyrB-gyrA', 'gyrA-Rv0007', 'iniB-iniA-iniC', 'iniB-iniA', 'iniC-lpqJ',
         'rpoB-rpoC', 'fabG1-inhA', 'rrs-rrl', 'rrl-rrf', 'inhA-hemZ', 'katG-furA', 'kasA-kasB',
         'ahpC-ahpD', 'dfrA-thyA', 'alr-Rv3792', 'embA-embB', 'embB-Rv3796', 'menG-ethA', 'rpsA', 'eis',
         "oxyR'", 'acpM']

# Get training and phenotypic data
df_X = pd.read_csv("X_features_with_names.csv", index_col=0)
X = np.loadtxt("X_features.csv", delimiter=',')
alpha_matrix = np.loadtxt("alpha_matrix.csv", delimiter=',')
y_true = np.loadtxt("labels.csv", delimiter=',')

valid_snp_inds_all = np.squeeze(np.where((big_X == 1).sum(axis=0) >= 30))
X = big_X[:,valid_snp_inds_all]

# Train and get predictions
clf = get_wide_deep()
clf.fit(X, alpha_matrix, nb_epoch=100)
#clf_dom = K.Function(clf.inputs + [K.learning_phase()], clf.outputs)
y_pred_dom = clf_dom.predict(X)

# Create permutation distribution
def distribute(snp_data, y_pred):
    diff = np.zeros((num_permute, num_drugs), dtype=np.float64)
    for index in range(num_permute):
        shuffled = np.random.permutation(snp_data)
        prob_1_given_1_shuffle = y_pred[(np.where(shuffled == 1))]
        prob_1_given_0_shuffle = y_pred[(np.where(shuffled == 0))]
        diff[index] = (np.mean(prob_1_given_1_shuffle, axis=0) - np.mean(prob_1_given_0_shuffle, axis=0))
    return diff

# Used to store data
snp_data = np.zeros((num_drugs, len(valid_snp_inds_all), 4), dtype=np.object)
final_sig_snps = np.zeros((num_drugs, 4), dtype=np.object)

s = 0
# Get mutations, p-value from permutation test, and exact difference in probabilities
for snp in valid_snp_inds_all:
    X_curr = big_X[:,snp][big_X[:,snp] != 0.5]
    y_curr = y_pred_dom[big_X[:,snp] != 0.5]
    permute_data = distribute(X_curr, y_curr)
    for drug in range(num_drugs):
        prob_1_given_1 = y_curr[:,drug][(np.where(X_curr == 0))]
        prob_1_given_0 = y_curr[:,drug][(np.where(X_curr == 1))]
        diff_exact = np.mean(prob_1_given_1) - np.mean(prob_1_given_0)
        permute_data_drug = permute_data[:,drug]
        sig = ((permute_data_drug > np.abs(diff_exact)) |
               (permute_data_drug < -np.abs(diff_exact))).sum()
        p_value = float(sig) / len(permute_data)
        snp_data[drug, s, 0] = snp
        snp_data[drug, s, 1] = 'sensitive' if diff_exact < 0 else 'resistant'
        snp_data[drug, s, 2] = p_value
        snp_data[drug, s, 3] = diff_exact
    s += 1

# Get mutations that meet critical value
for drug in range(num_drugs):
    swapped = np.swapaxes(snp_data, 1, 2)
    snp_inds = np.where(swapped[drug][2] < crit_value / len(valid_snp_inds_all))
    final_sig_snps[drug, 0] = list(df_X.columns[list(swapped[drug][0][snp_inds])])
    final_sig_snps[drug, 1] = list(swapped[drug][1][snp_inds])
    final_sig_snps[drug, 2] = list(swapped[drug][2][snp_inds])
    final_sig_snps[drug, 3] = list(swapped[drug][3][snp_inds])

# Save significant features
outarr = np.zeros(num_drugs, dtype=np.object)
for i in range(num_drugs):
    outarr[i] = np.vstack((final_sig_snps[i]))
    np.savetxt('snpsF{drug}020318.csv'.format(drug=drugs[i]), np.transpose(outarr[i]), fmt='%s', delimiter=',',
               header='SNPs, S/R, p-value, (y=S|s=1) - (y=S|s=0)')

# Save valid feature names
derived_names_orig = list(df_X.columns.values[valid_snp_inds_all])
np.savetxt('feature_names020318.csv', derived_names_orig, fmt='%s', delimiter=',')
