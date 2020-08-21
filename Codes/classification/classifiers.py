import numpy as np
import os
import sys
import sys
import time
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn import svm
from sklearn.kernel_approximation import Nystroem
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
#from sklearn.feature_selection import chi2
#from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


biol_dir = "/hpcfs/home/da.martinez33/Biologia"
classifiers_folder = os.path.join(biol_dir, 'Codes', 'classification')
sys.path.append(classifiers_folder)

import train_nn
import train_nn_cpu

# decision_function(self, X) Apply decision function to an array of samples.
# fit(self, X, y) Fit the model according to the given training data and parameters.
# get_params(self[, deep]) Get parameters for this estimator.
# predict(self, X) Perform classification on an array of test vectors X.
# predict_log_proba(self, X) Return posterior probabilities of classification.
# predict_proba(self, X) Return posterior probabilities of classification.
# score(self, X, y[, sample_weight]) Return the mean accuracy on the given test data and labels.
# set_params(self, \*\*params) Set the parameters of this estimator.

def feature_sel_funct(model, train_data, train_labels, test_data, test_labels,
    fs_type="KBest"):
    """ Function to apply a feature selection method to clf features"""
    fs_results = []
    num_features = train_data.shape[1]
    if fs_type == "KBest":
        for i in range(num_features):
            fs = SelectKBest(f_classif, k=i+1)
            new_train_data = fs.fit_transform(train_data, train_labels)
            selectedindices = fs.get_support(indices=True)
            clf_model = model
            clf_model.fit(new_train_data, train_labels)
            score = clf_model.score(test_data[:,selectedindices], test_labels)
            dict_fs = {'num_fs' : i+1, 'fs_type' : fs_type, 'fs_indices' : selectedindices,
            'score' : score}
            fs_results.append(dict_fs)
    if fs_type == "RFE":
        for i in range(num_features):
            est = model
            fs = RFE(estimator=est, n_features_to_select=i+1, step=1)
            new_train_data = fs.fit_transform(train_data, train_labels)
            selectedindices = fs.get_support(indices=True)
            clf_model = model
            clf_model.fit(new_train_data, train_labels)
            score = clf_model.score(test_data[:,selectedindices], test_labels)
            dict_fs = {'num_fs' : i+1, 'fs_type' : fs_type, 'fs_indices' : selectedindices,
            'score' : score}
            fs_results.append(dict_fs)
    if fs_type == "SFM":
        fs = SelectFromModel(model)
        new_train_data = fs.fit_transform(train_data, train_labels)
        selectedindices = fs.get_support(indices=True)
        clf_model = model
        clf_model.fit(new_train_data, train_labels)
        score = clf_model.score(test_data[:,selectedindices], test_labels)
        dict_fs = {'num_fs' : len(selectedindices), 'fs_type' : fs_type, 'fs_indices' : selectedindices,
        'score' : score}
        fs_results.append(dict_fs)
    return fs_results

def num_gamma(data,gamma_type='scale'):
    if gamma_type == 'scale':
        gamma_value = 1/(data.shape[1] * data.var())
    elif gamma_type == 'auto':
        gamma_value = 1/(data.shape[1])
    else:
        if isinstance(gamma_type,float):
            gamma_value = gamma_type
        else:
            raise ValueError('Select a correct gamma type or gamma value')
    return gamma_value

def get_train_results(model, test_data, test_labels, is_svc=False):
    preds = model.predict(test_data)
    mean_acc = model.score(test_data, test_labels)
    if is_svc:
        dec_funct = model.decision_function(test_data)
        return [test_labels, preds, dec_funct, mean_acc]
    else:
        probas = model.predict_proba(test_data)
        return [test_labels, preds, probas, mean_acc]

def classes_proportion(train_labels, test_labels):
    classes = list(set(train_labels))
    props_train = []
    props_test = []

    for c in classes:
        num_elems_train = len(np.where(train_labels==c)[0])
        num_elems_test = len(np.where(test_labels==c)[0])
        props_train.append(num_elems_train/len(train_labels))
        props_test.append(num_elems_test/len(test_labels))
    return [props_train, props_test]

# 3 Clasificadores: QDA, RF, SVC
def qda_classif(train_data, train_labels, test_data, test_labels, fs_type="KBest"):
    results = {}
    clf = QDA()
    if fs_type is not None:
        fs_res = feature_sel_funct(clf, train_data, train_labels, test_data, test_labels, fs_type=fs_type)
    clf.fit(train_data, train_labels)
    qda_results = get_train_results(clf, test_data, test_labels)
    props = classes_proportion(train_labels, test_labels)
    results['model'] = clf
    results['clf_results'] = qda_results[:4]
    results['props'] = props
    if fs_type is not None:
        results['fs_results'] = qda_results
    return results

def rf_classif(train_data, train_labels, test_data, test_labels, n_trees=100, boots=False, fs_type="KBest"):
    results = {}
    clf = RFC(n_estimators=n_trees, bootstrap=boots, max_features="sqrt")
    if fs_type is not None:
        fs_res = feature_sel_funct(clf, train_data, train_labels, test_data, test_labels, fs_type=fs_type)
    clf.fit(train_data, train_labels)
    rf_results = get_train_results(clf, test_data, test_labels)
    props = classes_proportion(train_labels, test_labels)
    results['model'] = clf
    results['clf_results'] = rf_results[:4]
    results['props'] = props
    if fs_type is not None:
        results['fs_results'] = fs_res
    return results

def svc_classif(train_data, train_labels, test_data, test_labels,
    c=1.0, kernel_type='linear', gamma_value='scale', degree_value = 3, fs_type="KBest"):
    results = {}
    if kernel_type == 'linear':
        clf = svm.LinearSVC(C=c)
        if fs_type is not None:
            fs_res = feature_sel_funct(clf, train_data, train_labels, test_data, test_labels, fs_type=fs_type)
        clf.fit(train_data, train_labels)
        svc_results = get_train_results(clf, test_data, test_labels, is_svc=True)
    else:
        #clf = svm.SVC(C=c, kernel=kernel_type, gamma=gamma_value, cache_size=1000)
        #clf = svm.LinearSVC(C=c)
        clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
        feature_map_nystroem = Nystroem(kernel=kernel_type,
            degree=degree_value,gamma=num_gamma(train_data,gamma_value),n_components=100)
        train_data_transform = feature_map_nystroem.fit_transform(train_data)
        print('data transformed to kernel type: {}\n'.format(kernel_type))
        test_data_transform = feature_map_nystroem.fit_transform(test_data)
        if fs_type is not None:
            fs_res = feature_sel_funct(clf, train_data_transform, train_labels, test_data_transform,
                test_labels, fs_type=fs_type)
        clf.fit(train_data_transform, train_labels)
        svc_results = get_train_results(clf, test_data_transform, test_labels, is_svc=True)
    props = classes_proportion(train_labels, test_labels)
    results['model'] = clf
    results['clf_results'] = svc_results[:4]
    results['props'] = props
    if fs_type is not None:
        results['fs_results'] = fs_res
    return results

def cnn_classif(train_data, train_labels, test_data, test_labels, num_fold=0, Net_type='mnist_net',
    gpu=True, fs_type="KBest"):
    #results = train_nn.main(train_data, train_labels, test_data, test_labels)
    results = {}
    if gpu:
        cnn_results = train_nn.main(train_data, train_labels, test_data, test_labels, num_fold, Net_type)
    else:
        cnn_results = train_nn_cpu.main(train_data, train_labels, test_data, test_labels, num_fold, Net_type)
    props = classes_proportion(train_labels, test_labels)
    results['model'] = cnn_results[0]
    results['clf_results'] = cnn_results[1:5]
    results['props'] = props
    return results
