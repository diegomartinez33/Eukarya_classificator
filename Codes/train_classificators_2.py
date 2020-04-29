import numpy as np
import os
import sys
import sys
import time

# Code to append libraries
biol_dir = "/hpcfs/home/da.martinez33/Biologia"
file_manage_folder = os.path.join(biol_dir,'Codes','file_management')
data_partition_folder = os.path.join(biol_dir,'Codes','data_partition')
metrics_folder = os.path.join(biol_dir,'Codes','metrics')
sys.path.append(file_manage_folder)
sys.path.append(data_partition_folder)
sys.path.append(metrics_folder)

from file_management import rws_files as rws
from data_partition import cross_validation as cv
from metrics import get_metrics_binary as gmb

## directories
biol_dir = "/hpcfs/home/da.martinez33/Biologia"
data_dir = os.path.join(biol_dir,"Codes","counts")
classif_dir = os.path.join(biol_dir,"Data","classification")
partitions_dir = os.path.join(classif_dir,"data_partitions")
models_dir = os.path.join(classif_dir,"models")
results_dir = os.path.join(biol_dir,"results","training")

def get_metrics(train_results, type_crossval, type_classif, **kwargs):
	#exp_folder = "{}_{}_cv".format(type_crossval,type_classif)
	if type_classif == 'rf':
		exp_folder = "{}_{}_{}_{}_cv".format(type_crossval,type_classif,
			kwargs["n_trees"], kwargs["boots"])
	elif type_classif == 'svc':
		exp_folder = "{}_{}_{}_{}_cv".format(type_crossval,type_classif,
			kwargs["c"], kwargs["kernel_type"], kwargs["gamma_value"])
	else:
		exp_folder = "{}_{}_cv".format(type_crossval,type_classif)

	savefolder = os.path.join(results_dir,exp_folder)
	if os.path.isdir(savefolder) != True:
		os.mkdir(savefolder)
	gmb.ACC_score(train_results, savefolder + '/ACC_score.txt')
	gmb.classif_report(train_results, savePath=savefolder + '/classif_report.txt')
	gmb.get_prf(train_results, savefolder + '/get_prf.txt')
	gmb.mcc(train_results, savefolder + '/mcc.txt')
	gmb.p_r_curve_cv(train_results, savefolder + '/P_R_curve_cv.png')
	gmb.ROC_curve(train_results, savefolder + '/ROC_curve_cv.png')


# Código para entrenar con los diferentes clasificadores y métodos de crossvalidación
def train_model(type_crossval="k-fold", type_classif="qda"):
	""" Function to train a classifier depending on:
	type_crossval: Type of cross validation iterator
	- "k-fold": make just homogeneus k-fold
	- "shuffleSplit": Add a shuffle representation of k-fold
	- "groups_k-fold": Generates cross-validation depending on some groups
	- "L_P_Groups_out": Leave P groups Out
	tyoe_classif: Type of classifier use to train the model
	- "qda"
	- "rf 
	- "svc" """
	split_type="general" #"all" or "general"
	test_percent=0.2
	type_labeling="per_group" #"per_group" or "per_spp"
	splitted_data_file = os.path.join(partitions_dir,
		"split_{}_{}_{}.pkl".format(split_type,str(test_percent),type_labeling))
	train_dict, test_dict = rws.loadData(splitted_data_file)

	train_data = train_dict['data']
	train_labels = train_dict['labels']
	s_labels = train_dict['s_labs']
	train_grp_animals = train_dict['train_grps']

	cross_val_test_size=0.2

	if type_classif == 'qda':
		kwargs = {}

	elif type_classif == 'rf':
		kwargs = {"n_trees" : 10, "boots" : False}

	elif type_classif == 'svc':
		kwargs = {"c" : 1.0, "kernel_type" : 'linear', "gamma_value" : 'scale'}

	else:
		raise ValueError("Select an aproppriate type of classifier")

	if split_type == 'all':

		num_splits = 5
		if type_crossval == 'k-fold':
			train_results = cv.k_fold_iter(train_data, train_labels, num_ss=num_splits, 
				clf=type_classif, **kwargs)

		elif type_crossval == 'shuffleSplit':
			test_size = cross_val_test_size
			train_results = cv.shuffle_split_iter(train_data, train_labels, num_ss=5, 
				t_size=test_size, clf=type_classif, **kwargs)

		else:
			raise ValueError("If split_type: 'all'," +
				" you have to choose type_crossval: k-fold or shuffleSplit")

	if split_type == "general":

		s_labels = train_dict['s_labs']
		train_grps_animals = train_dict['train_grps']
		n_groups = round(((1-test_percent) * 10)/2) 
		if type_crossval == 'groups_k-fold':
			train_results = cv.groups_k_fold_iter(train_data, train_labels, s_labels, 
				train_grp_animals, num_groups=n_groups, clf=type_classif, **kwargs)

		elif type_crossval == 'L_P_Groups_out':
			n_groups = 2
			train_results = leave_P_out_iter(train_data, train_labels, s_labels, train_grp_animals, 
				num_groups=n_groups, clf=type_classif, **kwargs)

		else:
			raise ValueError("If split_type: 'general'," + 
			 " you have to choose type_crossval: groups_k-fold or L_P_Groups_out")

	print(len(train_results))
	print(train_results[0][1].shape)
	print(train_results[0][2].shape)
	print(train_results[0][2])
	if len(train_results[0]) == 5:
		print(train_results[0][3].shape)
		print(train_results[0][3])
		print(train_results[0][4])
	else:
		print(train_results[0][3])

	acc_values = np.zeros(len(train_results))
	for i in range(len(train_results)):
		if len(train_results[0]) == 5:
			acc_values[i] = train_results[i][4]
		else:
			acc_values[i] = train_results[i][3]

	print(np.mean(acc_values))

	get_metrics(train_results, type_crossval, type_classif, **kwargs)


if __name__ == '__main__':
	#train_model(type_crossval="k-fold", type_classif="svc")
	train_model(sys.argv[1], sys.argv[2])