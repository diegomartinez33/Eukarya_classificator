import numpy as np
import os
import sys
import sys
import time
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn import svm

# decision_function(self, X) Apply decision function to an array of samples.
# fit(self, X, y) Fit the model according to the given training data and parameters.
# get_params(self[, deep]) Get parameters for this estimator.
# predict(self, X) Perform classification on an array of test vectors X.
# predict_log_proba(self, X) Return posterior probabilities of classification.
# predict_proba(self, X) Return posterior probabilities of classification.
# score(self, X, y[, sample_weight]) Return the mean accuracy on the given test data and labels.
# set_params(self, \*\*params) Set the parameters of this estimator.
def get_train_results(model, test_data, test_labels, is_svc=False):
	preds = model.predict(test_data)
	mean_acc = model.score(test_data, test_labels)
	if is_svc:
		return [test_labels, preds, mean_acc]
	else:
		probas = model.predict_proba(test_data)
		return [test_labels, preds, probas, mean_acc]

# 3 Clasificadores: QDA, RF, SVC
def qda_classif(train_data, train_labels, test_data, test_labels):
	clf = QDA()
	clf.fit(train_data, train_labels)
	results = get_train_results(clf, test_data, test_labels)
	return [clf, results[0], results[1], results[2], results[3]]

def rf_classif(train_data, train_labels, test_data, test_labels, 
	n_trees=100, boots=False):
	clf = RFC(n_estimators=n_trees, bootstrap=boots, max_features="sqrt")
	clf.fit(train_data, train_labels)
	results = get_train_results(clf, test_data, test_labels)
	return [clf, results[0], results[1], results[2], results[3]]

def svc_classif(train_data, train_labels, test_data, test_labels,
	c=1.0, kernel_type='linear', gamma_value='scale'):
	clf = svm.SVC(C=c, kernel=kernel_type, gamma=gamma_value, probability=True)
	clf.fit(train_data, train_labels)
	results = get_train_results(clf, test_data, test_labels, is_svc=True)
	return [clf, results[0], results[1], results[2], results[3]]
