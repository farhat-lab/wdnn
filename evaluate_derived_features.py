#from helpers import *
#from models import *
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold, StratifiedKFold
import keras.backend as K
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# Definitions
repeats = 5
cv_splits = 10
num_drugs = 11
drugs = ['rif', 'inh', 'pza', 'emb', 'str', 'cip', 'cap', 'amk', 'moxi', 'oflx', 'kan']


data_dir = 'training_data_102618/'
# Data
X = np.loadtxt(data_dir + 'X_features.csv', delimiter=',')

alpha_matrix = np.loadtxt(data_dir + 'alpha_matrix.csv', delimiter=',')
y = np.loadtxt(data_dir + 'labels.csv', delimiter=',')

# Mutations unavailable through subset of isolates that underwent targeted sequencing
X[X == -1] = 0.5

# Get mutations that appear in at least 30 isolates
sufficient_inds = np.squeeze(np.where((X == 1).sum(axis=0) >= 30))
X = X[:,sufficient_inds]

## Derived features are last 56 columns
raw_features_end = X.shape[1] - 56
X = X[:,:raw_features_end]

column_names = ['Algorithm','Drug','AUC','AUC_PR']
results = pd.DataFrame(columns=column_names)
results_index = 0
for r in range(repeats):
    cross_val_split = KFold(n_splits=cv_splits, shuffle=True)
    for train, val in cross_val_split.split(X):
        X_train = X[train]
        X_val = X[val]
        y_train = y[train]
        y_val = y[val]
        #------- Train the wide and deep neural network ------#
        wdnn = get_wide_deep_raw_features()
        wdnn.fit(X_train, alpha_matrix[train], epochs=100, verbose=False, validation_data=[X_val,alpha_matrix[val]])
        mc_dropout = K.Function(wdnn.inputs + [K.learning_phase()], wdnn.outputs)
        #wdnn_probs = ensemble(X_val, y_val, mc_dropout)
        wdnn_probs = wdnn.predict(X_val)
        for i, drug in enumerate(drugs):
            non_missing_val = np.where(y_val[:,i] != -1)[0]
            auc_y = np.reshape(y_val[non_missing_val,i],(len(non_missing_val), 1))
            auc_preds = np.reshape(wdnn_probs[non_missing_val,i],(len(non_missing_val), 1))
            val_auc = roc_auc_score(auc_y, auc_preds)
            val_auc_pr = average_precision_score(1-y_val[non_missing_val,i], 1-wdnn_probs[non_missing_val,i])
            results.loc[results_index] = ['WDNN Raw Features',drug,val_auc,val_auc_pr]
            print drug + '\t' + str(val_auc) + '\t' + str(val_auc_pr)
            results_index += 1

for r in range(repeats):
    for i, drug in enumerate(drugs):
        y_drug = y[:, i]
        # Disregard rows for which no resistance data exists
        y_non_missing = y_drug[y_drug != -1]
        X_non_missing = X[y_drug != -1]
        cross_val_split = KFold(n_splits=cv_splits, shuffle=True)
        for train, val in cross_val_split.split(X_non_missing):
            X_train = X_non_missing[train]
            X_val = X_non_missing[val]
            y_train = y_non_missing[train]
            y_val = y_non_missing[val]
            # Train and predict on random forest classifier
            random_forest = RandomForestClassifier(n_estimators=1000, max_features='auto', min_samples_leaf=0.002)
            random_forest.fit(X_train, y_train)
            pred_rf = random_forest.predict_proba(X_val)
            # Get AUC of drug for RF
            rf_auc = roc_auc_score(y_val, pred_rf[:,1])
            rf_auc_pr = average_precision_score(1-y_val, 1-pred_rf[:,1])
            results.loc[results_index] = ['Random Forest Raw Features', drug, rf_auc, rf_auc_pr]
            results_index += 1
            # Train and predict on regularized logistic regression model
            log_reg = LogisticRegression(penalty='l2')
            Cs = np.logspace(-5, 5, 10)
            estimator = GridSearchCV(estimator=log_reg, param_grid={'C': Cs}, cv=5, scoring='roc_auc')
            estimator.fit(X_train, y_train)
            pred_lm = estimator.predict_proba(X_val)
            lm_auc = roc_auc_score(y_val, pred_lm[:,1])
            lm_auc_pr = average_precision_score(1-y_val, 1-pred_lm[:, 1])
            results.loc[results_index] = ['Logistic Regression Raw Features', drug, lm_auc, lm_auc_pr]
            results_index += 1


results.to_csv('results_raw_features_rf_lm.csv',index=False)

