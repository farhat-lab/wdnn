import timeit
import pandas as pd

n_iter_time = 1000
cv_splits = 5

wdnn_time = 0
cv1 = KFold(n_splits=cv_splits, shuffle=True, random_state=100)
for train, val in cv1.split(X):
	clf_m = get_wide_deep()
	clf_m.fit(X[train], alpha_matrix[train], nb_epoch=50)
	wdnn_time += timeit.timeit('clf_m.predict(X[val])', globals=globals(), number=n_iter_time)

wdnn_time /= cv_splits * n_iter_time

lr_time = 0
for i, drug in enumerate(drugs):
	if drugs[i] != 'cip':
	    # Label data for current drug
	    y_true_drug = y_true[:,i]
	    # Disregard rows for which no resistance data exists
	    y_true_small = y_true_drug[y_true_drug != -1]
	    X_small = X[y_true_drug != -1]
	    # Stratified cross-validation split
	    cv3 = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=100)
	    fold = 0
	    for train, val in cv3.split(X_small, y_true_small):
	        # Training and validation label data
	        y_train = y_true_small[train]
	        y_val = y_true_small[val]
	        # Train and predict on regularized logistic regression model
	        log_reg = LogisticRegression(penalty='l1', solver='liblinear')
	        Cs = np.logspace(-5, 5, 10)
	        estimator = GridSearchCV(estimator=log_reg, param_grid={'C': Cs}, cv=5, scoring='roc_auc')
	        estimator.fit(X_small[train], y_train)
	        lr_time += timeit.timeit('estimator.predict_proba(X[val])', globals=globals(), number=n_iter_time)

lr_time /= cv_splits * n_iter_time




########## Time on validation set
wdnn_time = 0
cv1 = KFold(n_splits=cv_splits, shuffle=True, random_state=100)
wdnn = get_wide_deep()
wdnn.fit(X, alpha_matrix, epochs=50)
wdnn_time += timeit.timeit('wdnn.predict(X_test)', globals=globals(), number=n_iter_time) / n_iter_time

lr_time = 0
# Get performance data for RF and LR
for i, drug in enumerate(drugs):
    if drug != 'cip':
        y_drug = y_true[:, i]
        # Disregard rows for which no resistance data exists
        y_non_missing = y_drug[y_drug != -1]
        X_non_missing = X[y_drug != -1]
        X_train = X_non_missing
        y_train = y_non_missing
        # Train and predict on regularized logistic regression model
        log_reg = LogisticRegression(penalty='l2', solver='liblinear')
        Cs = np.logspace(-5, 5, 10)
        estimator = GridSearchCV(estimator=log_reg, param_grid={'C': Cs}, cv=5, scoring='roc_auc')
        estimator.fit(X_train, y_train)
        lr_time += timeit.timeit('estimator.predict_proba(X_test)', globals=globals(), number=n_iter_time) / n_iter_time
