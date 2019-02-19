import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_selection import SelectFromModel

# requires X_features_with_names.csv as df_X
# requires X_features as X

data_dir = 'raw_data/'

# Get training data
X = np.loadtxt(data_dir + 'X_features.csv', delimiter=',')
#alpha_matrix = np.loadtxt(data_dir + 'alpha_matrix.csv', delimiter=',')
y_true = np.loadtxt(data_dir + 'labels.csv', delimiter=',')
df_X = pd.read_csv(data_dir + 'X_features_with_names.csv', index_col=0)

drugs = ['rif', 'inh', 'pza', 'emb', 'str', 'cip', 'cap', 'amk', 'moxi', 'oflx', 'kan']

# Get independent test set to numpy arrays
#df_X_test = pd.read_csv(data_dir + "validation_data.csv")
#df_y_test = pd.read_csv(data_dir + "validation_data_pheno.csv")

sufficient_inds = np.squeeze(np.where((X == 1).sum(axis=0) >= 30))
X = X[:,sufficient_inds]
small_df_X = df_X[df_X.columns[sufficient_inds]]

#X_test = df_X_test.as_matrix()
#y_test = df_y_test.as_matrix()

#column_names = ['Algorithm','Drug','AUC', 'Sensitivity', "Specificity", "Threshold"]
#results = pd.DataFrame(columns=column_names)
#results_index = 0
num_iter = 10000
odds_ratios = np.zeros((11, num_iter, X.shape[1]), dtype=np.dtype('float16'))
# Get performance data for RF and LR
for i, drug in enumerate(drugs):
    if drug != 'cip':
        y_drug = y_true[:, i]
        # Disregard rows for which no resistance data exists
        y_non_missing = y_drug[y_drug != -1]
        X_non_missing = X[y_drug != -1]
        X_train = X_non_missing
        y_train = y_non_missing
        # Disregard rows in test set where no resistance data exists
        #y_test_non_missing = y_test[y_test[:,i] != -1,i]
        #X_test_non_missing = X_test[y_test[:, i] != -1, :]
        # Train on regularized logistic regression model with all 
        for j in range(num_iter):
            bootstrap_indx = np.random.choice(range(y_train.shape[0]), size=y_train.shape[0], replace=True)
            X_temp = X_train[bootstrap_indx]
            y_temp = y_train[bootstrap_indx]
            log_reg = LogisticRegression(penalty='l2', solver='liblinear')
            Cs = np.logspace(-5, 5, 10)
            estimator = GridSearchCV(estimator=log_reg, param_grid={'C': Cs}, cv=5, scoring='roc_auc')
            estimator.fit(X_temp, y_temp)
            odds_ratios[i][j] = np.exp(estimator.best_estimator_.coef_)
            if j % 100 == 0:
                print(drug + str(j))
        


### Get LR important features CSV
num_drugs = 11
snps_list = np.array(small_df_X.columns.tolist())
alpha_star = 0.05 / X.shape[1]
for i in range(num_drugs):
    column_names = ['SNPs','S/R','Lower CI','Higher CI']
    results = pd.DataFrame(columns=column_names)
    results_index = 0
    upper = np.percentile(odds_ratios[i], q=1-alpha_star/2, axis=0)
    lower = np.percentile(odds_ratios[i], q=alpha_star/2, axis=0)
    resistant_indx = np.where(upper < 1)[0]
    susceptible_indx = np.where(lower > 1)[0]
    for j in resistant_indx:
        results.loc[results_index] = [snps_list[j], 'resistant', lower[j], upper[j]]
        results_index += 1
    for j in susceptible_indx:
        results.loc[results_index] = [snps_list[j], 'sensitive', lower[j], upper[j]]
        results_index += 1
    results.to_csv('lr_snps_111218/lr_snps_111218_'+drugs[i]+'.csv',index=False)


