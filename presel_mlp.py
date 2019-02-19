#from helpers import *
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from keras.layers import Dense, Dropout, Activation, Flatten, Input, BatchNormalization
from keras.models import Model
from keras.layers.convolutional import *
import keras.backend as K
from sklearn.metrics import roc_auc_score, average_precision_score

# Definitions
repeats = 5
cv_splits = 10
num_drugs = 11
drugs = ['rif', 'inh', 'pza', 'emb', 'str', 'cip', 'cap', 'amk', 'moxi', 'oflx', 'kan']

# Data

data_dir = '/mnt/raid1/TB_data/tb_data_050818/'
# Data
X = np.loadtxt(data_dir + 'X_features.csv', delimiter=',')
alpha_matrix = np.loadtxt(data_dir + 'alpha_matrix.csv', delimiter=',')
y_true = np.loadtxt(data_dir + 'labels.csv', delimiter=',')
df_X = pd.read_csv(data_dir + 'X_features_with_names.csv', index_col=0)

# Get mutations that appear in at least 30 isolates
sufficient_inds = np.squeeze(np.where((X == 1).sum(axis=0) >= 30))
X = X[:,sufficient_inds]

# Mutation names
derived_names_all = list(df_X.columns[sufficient_inds].values)

# Dictionary of mutations with gene keys
gene_dict = get_gene_dict(derived_names_all)

# Get pre-selected mutations known to be important to resistance/sensitivity for rifampicin
rif_names = ['rpoB', 'rpoB-rpoC']
rif_snps = []
for i in range(len(rif_names)):
    try:
        rif_snps += gene_dict[rif_names[i]]
    except KeyError:
        print(rif_names[i])
        continue

# Get pre-selected mutations known to be important to resistance/sensitivity for isoniazid
inh_names = ['ahpC', 'embB', 'inhA', 'iniA', 'iniB', 'iniC', 'kasA', 'katG', 'fabG1', 'ndh', "oxyR'",
            'iniA-iniB-iniC', 'iniA-iniB', 'iniC-lpqJ', 'fabG1-inhA', 'hemZ-inhA', 'kasA-kasB',
            'ahpC-ahpD', 'embA-embB', 'Rv3796-embB']
inh_snps = []
for i in range(len(inh_names)):
    try:
        inh_snps += gene_dict[inh_names[i]]
    except KeyError:
        print(inh_names[i])
        continue

# Get pre-selected mutations known to be important to resistance/sensitivity for pyrazinamide
pza_names = ['pncA', 'rpsA']
pza_snps = []
for i in range(len(pza_names)):
    try:
        pza_snps += gene_dict[pza_names[i]]
    except KeyError:
        print(pza_names[i])
        continue

# Get pre-selected mutations known to be important to resistance/sensitivity for ethambutol
emb_names = ['embA', 'embB', 'embC', 'iniA', 'iniB', 'iniC', 'iniA-iniB', 'iniA-iniB-iniC', 'iniC-lpqJ',
            'embA-embB', 'Rv3796-embB']
emb_snps = []
for i in range(len(emb_names)):
    try:
        emb_snps += gene_dict[emb_names[i]]
    except KeyError:
        print(emb_names[i])
        continue

# Get pre-selected mutations known to be important to resistance/sensitivity for streptomycin
str_names = ['gid', 'murA-rrs', 'rpsL', 'rrl', 'rrs', 'rrl-rrs', 'rrf-rrl']
str_snps = []
for i in range(len(str_names)):
    try:
        str_snps += gene_dict[str_names[i]]
    except KeyError:
        print(str_names[i])
        continue

# Get pre-selected mutations known to be important to resistance/sensitivity for ciprofloxacin
cip_names = ['gyrA', 'gyrB', 'gyrA-gyrB', 'Rv0007-gyrA']
cip_snps = []
for i in range(len(cip_names)):
    try:
        cip_snps += gene_dict[cip_names[i]]
    except KeyError:
        print(cip_names[i])
        continue

# Get pre-selected mutations known to be important to resistance/sensitivity for capreomycin
cap_names = ['murA-rrs', 'rrl', 'rrs', 'rrl-rrs', 'rrf-rrl', 'tlyA']
cap_snps = []
for i in range(len(cap_names)):
    try:
        cap_snps += gene_dict[cap_names[i]]
    except KeyError:
        print(cap_names[i])
        continue

# Get pre-selected mutations known to be important to resistance/sensitivity for amikacin
amk_names = ['murA-rrs', 'rrl', 'rrs', 'rrl-rrs', 'rrf-rrl']
amk_snps = []
for i in range(len(amk_names)):
    try:
        amk_snps += gene_dict[amk_names[i]]
    except KeyError:
        print(amk_names[i])
        continue

# Get pre-selected mutations known to be important to resistance/sensitivity for moxifloxacin
moxi_names = ['gyrA', 'gyrB', 'gyrA-gyrB', 'Rv0007-gyrA']
moxi_snps = []
for i in range(len(moxi_names)):
    try:
        moxi_snps += gene_dict[moxi_names[i]]
    except KeyError:
        print(moxi_names[i])
        continue

# Get pre-selected mutations known to be important to resistance/sensitivity for ofloxacin
oflx_snps = moxi_snps

# Get pre-selected mutations known to be important to resistance/sensitivity for kanamycin
kan_names = ['murA-rrs', 'rrl', 'rrs', 'rrl-rrs', 'rrf-rrl', 'eis']
kan_snps = []
for i in range(len(kan_names)):
    try:
        kan_snps += gene_dict[kan_names[i]]
    except KeyError:
        print(kan_names[i])
        continue

# Full array of all selected mutations
num_snp_indiv = [rif_snps, inh_snps, pza_snps, emb_snps, str_snps,
                 cip_snps, cap_snps, amk_snps, moxi_snps, oflx_snps, kan_snps]

output_dir = 'intermediate_data_102618/'
# Save preselected mutations per drug
np.savetxt(output_dir + "rif_snps.csv", rif_snps, delimiter=",", fmt="%s")
np.savetxt(output_dir + "inh_snps.csv", inh_snps, delimiter=",", fmt="%s")
np.savetxt(output_dir + "pza_snps.csv", pza_snps, delimiter=",", fmt="%s")
np.savetxt(output_dir + "emb_snps.csv", emb_snps, delimiter=",", fmt="%s")
np.savetxt(output_dir + "str_snps.csv", str_snps, delimiter=",", fmt="%s")
np.savetxt(output_dir + "cap_snps.csv", cap_snps, delimiter=",", fmt="%s")
np.savetxt(output_dir + "amk_snps.csv", amk_snps, delimiter=",", fmt="%s")
np.savetxt(output_dir + "moxi_snps.csv", moxi_snps, delimiter=",", fmt="%s")
np.savetxt(output_dir + "oflx_snps.csv", oflx_snps, delimiter=",", fmt="%s")
np.savetxt(output_dir + "kan_snps.csv", kan_snps, delimiter=",", fmt="%s")

# Store AUC, specificity, and sensitivity for pre-selected mutations WDNN
auc_strat_data, spec_strat_data, sens_strat_data = (make_data_array() for _ in range(3))

strat_thresholds = np.zeros((11, 5), dtype=np.float)

column_names = ['Algorithm','Drug','AUC','AUC_PR']
results = pd.DataFrame(columns=column_names)
results_index = 0

# Single Task WDNN preslected
for r in range(repeats):
    print(str(i))
    for i, drug in enumerate(drugs):
        # Get feature and label data for current drug
        X = df_X[num_snp_indiv[i]].as_matrix()
        y_true_drug = y_true[:,i]
        # Disregard rows for which no resistance data exists
        y_true_small = y_true_drug[y_true_drug != -1]
        X_small = X[y_true_drug != -1]
        # Stratified cross-validation split
        cv3 = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=100)
        fold = 0
        for train, val in cv3.split(X_small, y_true_small):
            X_train = X_small[train]
            X_val = X_small[val]
            # Training and validation label data
            y_train = y_true_small[train]
            y_val = y_true_small[val]
            # Train and predict
            wdnn_pre = get_wide_deep_preselect(len(num_snp_indiv[i]))
            wdnn_pre.fit(X_small[train], y_train, nb_epoch=100,verbose=True)
            wdnn_pre_preds = wdnn_pre.predict(X_val)
            # Get AUC, specificity, and sensitivity of drug for single task WDNN
            wdnn_pre_auc = roc_auc_score(y_val.reshape(len(y_val), 1), wdnn_pre_preds.reshape((len(wdnn_pre_preds), 1)))
            wdnn_pre_auc_pr = average_precision_score(1-y_val.reshape(len(y_val), 1), 1-wdnn_pre_preds.reshape((len(wdnn_pre_preds), 1)))
            results.loc[results_index] = ['WDNN Single Task (Select Mutations)', drug, wdnn_pre_auc, wdnn_pre_auc_pr]
            results_index += 1

# Random Forest and Logistic Regression
for r in range(repeats):
    print(str(i))
    for i, drug in enumerate(drugs):
        # Get feature and label data for current drug
        X = df_X[num_snp_indiv[i]].as_matrix()
        y_true_drug = y_true[:,i]
        # Disregard rows for which no resistance data exists
        y_true_small = y_true_drug[y_true_drug != -1]
        X_small = X[y_true_drug != -1]
        # Stratified cross-validation split
        cv3 = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=100)
        fold = 0
        for train, val in cv3.split(X_small, y_true_small):
            X_train = X_small[train]
            X_val = X_small[val]
            # Training and validation label data
            y_train = y_true_small[train]
            y_val = y_true_small[val]
            # Train and predict on random forest classifier
            random_forest = RandomForestClassifier(n_estimators=1000, max_features='auto', min_samples_leaf=0.002)
            random_forest.fit(X_train, y_train)
            pred_rf = random_forest.predict_proba(X_val)
            # Get AUC of drug for RF
            rf_auc = roc_auc_score(y_val, pred_rf[:,1])
            rf_auc_pr = average_precision_score(1-y_val, 1-pred_rf[:,1])
            results.loc[results_index] = ['Random Forest (Select Mutations)', drug, rf_auc, rf_auc_pr]
            results_index += 1
            # Train and predict on regularized logistic regression model
            log_reg = LogisticRegression(penalty='l2', solver='liblinear')
            Cs = np.logspace(-5, 5, 10)
            estimator = GridSearchCV(estimator=log_reg, param_grid={'C': Cs}, cv=5, scoring='roc_auc')
            estimator.fit(X_train, y_train)
            pred_lm = estimator.predict_proba(X_val)
            lm_auc = roc_auc_score(y_val, pred_lm[:,1])
            lm_auc_pr = average_precision_score(1-y_val, 1-pred_lm[:, 1])
            results.loc[results_index] = ['Logistic Regression (Select Mutations)', drug, lm_auc, lm_auc_pr]
            results_index += 1
            
results.to_csv('raw_results_102618/results_select_snps_rf_lm.csv',index=False)



