import numpy as np
import os
import sys
import sys
import time
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import LeavePGroupsOut
from sklearn.model_selection import ShuffleSplit

# Code for file managemente library
biol_dir = "/hpcfs/home/da.martinez33/Biologia"
data_partition_folder = os.path.join(biol_dir,'Codes','data_partition')
classifiers_folder = os.path.join(biol_dir, 'Codes', 'classification')
sys.path.append(data_partition_folder)
sys.path.append(classifiers_folder)

from data_partition import labeling as lb
from classification.classifiers import qda_classif
from classification.classifiers import rf_classif
from classification.classifiers import svc_classif
from classification.classifiers import cnn_classif

def get_train_function(clf='qda'):
	if clf == 'qda':
		return qda_classif
	elif clf == 'rf':
		return rf_classif
	elif clf == 'svc':
		return svc_classif
	elif clf == 'cnn':
		return cnn_classif
	else:
		raise ValueError("Specify a correct type of classfier")

def get_iterations(data, cv_function, labs=None, groups_list=None):
	inter = []
	if groups_list is None:
		for train_idx, test_idx in cv_function.split(data):
			inter.append([train_idx, test_idx])
	else:
		for train_idx, test_idx in cv_function.split(data, labs, groups=groups_list):
			inter.append([train_idx, test_idx])
	return inter

def wrong_clf_data(test_data, test_labels, test_s_labels, test_ids, preds):
	wrong_idx = np.where(test_labels != preds)
	wrong_data = test_data[wrong_idx]
	wrong_s_labels = test_s_labels[wrong_idx]
	wrong_ids = test_ids[wrong_idx]
	print("wrong data shape: ", wrong_data.shape)

	return [wrong_data, wrong_s_labels, wrong_ids]


def get_train_cv_results(data, labels, s_labels, ids, cv_function, clf_funct,
	groups_list=None, cnn_clf=False, **kwargs):
	train_cv = []
	if groups_list is None:
		cont = 1
		for train_idx, test_idx in cv_function.split(data):
			print("Begin fold: ",cont)
			train_fold, test_fold = data[train_idx], data[test_idx]
			train_labels, test_labels = labels[train_idx], labels[test_idx]
			if cnn_clf:
				train_results = clf_funct(train_fold, train_labels, test_fold, 
				test_labels, num_fold=cont, **kwargs)
			else:
				train_results = clf_funct(train_fold, train_labels, test_fold, 
					test_labels, **kwargs)
			cont += 1
			### Wrong classification
			test_s_labels = np.array(s_labels)[test_idx]
			test_ids = ids[test_idx]
			#print(train_results)
			train_results['wrong_results']=wrong_clf_data(test_fold, test_labels, test_s_labels, 
				test_ids, train_results['clf_results'][1])

			train_cv.append(train_results)
	else:
		cont = 1
		for train_idx, test_idx in cv_function.split(data, labels, groups=groups_list):
			print("Begin fold: ",cont)
			train_fold, test_fold = data[train_idx], data[test_idx]
			train_labels, test_labels = labels[train_idx], labels[test_idx]
			if cnn_clf:
				train_results = clf_funct(train_fold, train_labels, test_fold, 
				test_labels, num_fold=cont, **kwargs)
			else:
				train_results = clf_funct(train_fold, train_labels, test_fold, 
					test_labels, **kwargs)
			cont += 1
			### Wrong classification
			test_s_labels = np.array(s_labels)[test_idx]
			test_ids = ids[test_idx]
			train_results['wrong_results']=wrong_clf_data(test_fold, test_labels, test_s_labels, 
				test_ids, train_results['clf_results'][1])
			
			train_cv.append(train_results)
	return train_cv

#TODO: Funcion diferente por cada iterador de crosvalidación
def k_fold_iter(data, labels, s_labels, ids, num_ss=5, clf=None, is_cnn=False,
    **kwargs):
	""" """
	kf = KFold(n_splits=num_ss,shuffle=True)
	if clf is None:
		interations = get_iterations(data, kf)
		return interations
	else:
		print("Recibe tipo de clasificador")
		clf_function = get_train_function(clf)
		train_cv = get_train_cv_results(data, labels, s_labels, ids, cv_function=kf, 
			clf_funct=clf_function, cnn_clf=is_cnn, **kwargs)
		return train_cv

def shuffle_split_iter(data, labels, s_labels, ids, num_ss=5, t_size=0.2, clf=None,
    is_cnn=False, **kwargs):

	sf = ShuffleSplit(n_splits=num_ss,test_size=0.2)
	if clf is None:
		interations = get_iterations(data, sf)
		return interations
	else:
		clf_function = get_train_function(clf)
		train_cv = get_train_cv_results(data, labels, s_labels, ids, cv_function=sf, 
			clf_funct=clf_function, cnn_clf=is_cnn, **kwargs)
		return train_cv

def groups_k_fold_iter(data, labels, s_labels, ids, train_grp_animals, num_groups=4,
    clf=None, is_cnn=False, **kwargs):
	"""Function to separate folds as groups with separated instances of each organism
	num_groups: number of groups in the train split"""
	gkf = GroupKFold(n_splits=num_groups)
	groups_list = lb.grouping_crossval(s_labels, ani_gps=train_grp_animals)
	if clf is None:
		interations = get_iterations(data, gkf, labels, groups_list)
		return interations
	else:
		clf_function = get_train_function(clf)
		train_cv = get_train_cv_results(data, labels, s_labels, ids, gkf, 
			clf_function, groups_list, cnn_clf=is_cnn, **kwargs)
		return train_cv


def leave_P_out_iter(data, labels, s_labels, ids, train_grp_animals, num_groups=2,
    clf=None, is_cnn=False, **kwargs):
	"""Function to separate folds as groups with separated instances of each organism
	num_groups: number of groups in the train split. Can be 2 or 3 only"""

	lpgo = LeavePGroupsOut(n_groups=num_groups)
	groups_list = lb.grouping_crossval(s_labels, ani_gps=train_grp_animals)
	if clf is None:
		interations = get_iterations(data, lpgo, labels, groups_list)
		return interations
	else:
		clf_function = get_train_function(clf)
		train_cv = get_train_cv_results(data, labels, s_labels, ids, lpgo, 
			clf_function, groups_list, cnn_clf=is_cnn, **kwargs)
		return train_cv