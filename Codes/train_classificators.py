import numpy as np
import os
import sys
import time

# Code to append libraries
biol_dir = "/hpcfs/home/da.martinez33/Biologia"
file_manage_folder = os.path.join(biol_dir,'Codes','file_management')
data_partition_folder = os.path.join(biol_dir,'Codes','data_partition')
#classifiers_folder = os.path.join(biol_dir, 'Codes', 'classification')
metrics_folder = os.path.join(biol_dir,'Codes','metrics')
sys.path.append(file_manage_folder)
sys.path.append(data_partition_folder)
sys.path.append(metrics_folder)
#sys.path.append(classifiers_folder)

from file_management import rws_files as rws
from data_partition import cross_validation as cv
from metrics import get_metrics_binary as gmb
from v_plots import get_viols_wrongs

## directories
biol_dir = "/hpcfs/home/da.martinez33/Biologia"
data_dir = os.path.join(biol_dir,"Codes","counts")
classif_dir = os.path.join(biol_dir,"Data","classification")
partitions_dir = os.path.join(classif_dir,"data_partitions")
models_dir = os.path.join(classif_dir,"models")
results_dir = os.path.join(biol_dir,"results","training")

others_cnn = '0.0001'

def get_metrics(train_results, type_crossval, type_classif, **kwargs):
	#exp_folder = "{}_{}_cv".format(type_crossval,type_classif)
	if type_classif == 'rf':
		exp_folder = "{}_{}_{}_{}_cv".format(type_crossval,type_classif,
			kwargs["n_trees"], kwargs["boots"])
	elif type_classif == 'svc':
		exp_folder = "{}_{}_{}_{}_{}_2_cv".format(type_crossval,type_classif,
			kwargs["c"], kwargs["kernel_type"], kwargs["gamma_value"])
	elif type_classif == 'cnn':
		exp_folder = "{}_{}_{}_cv".format(type_crossval,type_classif,others_cnn)
	else:
		exp_folder = "{}_{}_cv".format(type_crossval,type_classif)

	savefolder = os.path.join(results_dir,exp_folder)
	if os.path.isdir(savefolder) != True:
		os.mkdir(savefolder)

	for fold in range(len(train_results)):
		print("props: \n")
		propors = train_results[fold]['props']
		line_train = "train:\t"
		line_test = "test:\t"
		for i in range(len(propors[0])):
			line_train = line_train + "class {}: {:4f}\t".format(i,propors[0][i])
			line_test = line_test + "class {}: {:4f}\t".format(i,propors[1][i])
		line = line_train + "\n" + line_test + "\n"
		print(line)

		rws.write_results(os.path.join(savefolder, "class_props.txt"), "Fold {}\n".format(fold))
		rws.write_results(os.path.join(savefolder, "class_props.txt"), line)

	gmb.ACC_score(train_results, savefolder + '/ACC_score.txt')
	gmb.classif_report(train_results, savePath=savefolder + '/classif_report.txt')
	gmb.get_prf(train_results, savefolder + '/get_prf.txt')
	gmb.mcc(train_results, savefolder + '/mcc.txt')
	gmb.p_r_curve_cv(train_results, type_classif, savefolder + '/P_R_curve_cv.png')
	gmb.ROC_curve(train_results, type_classif, savefolder + '/ROC_curve_cv.png')

def distr_missclass_data(kmers, train_results, type_crossval, type_classif, **kwargs):
	if type_classif == 'rf':
		exp_folder = "{}_{}_{}_{}_cv".format(type_crossval,type_classif,
			kwargs["n_trees"], kwargs["boots"])
	elif type_classif == 'svc':
		exp_folder = "{}_{}_{}_{}_{}_cv".format(type_crossval,type_classif,
			kwargs["c"], kwargs["kernel_type"], kwargs["gamma_value"])
	elif type_classif == 'cnn':
		exp_folder = "{}_{}_{}_cv".format(type_crossval,type_classif,others_cnn)
	else:
		exp_folder = "{}_{}_cv".format(type_crossval,type_classif)

	for num_fold in range(len(train_results)):
		print("\n---------------- fold %d ---------------------\n" % num_fold)
		wrong_data = train_results[num_fold]['wrong_results'][0]
		wrong_s_labels = train_results[num_fold]['wrong_results'][1]
		save_file = os.path.join(exp_folder, "fold_{}".format(num_fold))
		get_viols_wrongs(wrong_data, wrong_s_labels, kmers, save_file, all_data=False)
		get_viols_wrongs(wrong_data, wrong_s_labels, kmers, save_file, all_data=True)

def fs_results_manage(train_results, type_classif, kmers):
	for num_fold in range(len(train_results)):
		print("\n---------------- fold %d ---------------------\n" % num_fold)
		fs_results = train_results[num_fold]['fs_results']
		max_score = 0
		best_num_features = 0
		fs_type=''
		selected_kmers = []
		print("fs_results length: ", len(fs_results))
		for dict_fs in fs_results:
			#dict_fs = {'num_fs' : len(selectedindices),
			#'fs_type' : fs_type,
			#'fs_indices' : selectedindices,
			#'score' : score}
			score = dict_fs['score']
			fs_type_2 = dict_fs['fs_type']
			sel_kmers = kmers[dict_fs['fs_indices']]
			print("\nscore: {}\tfs_type: {}\tselected_kmers{}\n".format(score, fs_type_2, sel_kmers))
			if dict_fs['score'] >= max_score:
				max_score = dict_fs['score']
				fs_type = dict_fs['fs_type']
				selected_kmers = kmers[dict_fs['fs_indices']]
				#print("\nscore: {}\tfs_type: {}\tselected_kmers{}\n".format(max_score, fs_type, selected_kmers))
			else:
				continue

		exp_folder = "{}_{}".format(type_classif, fs_type)
		savefolder = os.path.join(results_dir,"fs_results", exp_folder)
		if os.path.isdir(savefolder) != True:
			os.makedirs(savefolder)

		print("fs_results: \n")
		
		line = "Fold: {} \t type_classif: {}\t fs_type: {}\t max_score: {:4f}\t\n".format(num_fold,
			type_classif, fs_type, max_score)
		line_2 = "kmers: {}\n\n".format(selected_kmers)
		print(line,line_2)

		rws.write_results(os.path.join(savefolder, "fs_results.txt"), line + line_2)

# Código para entrenar con los diferentes clasificadores y métodos de crossvalidación
def train_model(type_crossval="k-fold", type_classif="qda"):
	""" Function to train a classifier depending on:
	type_crossval: Type of cross validation iterator
	- "k-fold": make just homogeneus k-fold
	- "shuffleSplit": Add a shuffle representation of k-fold
	- "groups_k-fold": Generates cross-validation depending on some groups
	- "L_P_Groups_out": Leave P groups Out
	- "final_test": training with all train data set and test in final test set
	type_classif: Type of classifier use to train the model
	- "qda"
	- "rf 
	- "svc" 
	- "cnn """
	split_type="all" #"all" or "general"
	test_percent=0.2
	type_labeling="per_group" #"per_group" or "per_spp"
	splitted_data_file = os.path.join(partitions_dir,
		"split_{}_{}_{}.pkl".format(split_type,str(test_percent),type_labeling))
	train_dict, test_dict = rws.loadData(splitted_data_file)

	train_data = train_dict['data']
	train_labels = train_dict['labels']
	train_ids = train_dict['ids']
	s_labels = train_dict['s_labs']
	kmers = train_dict['kmers']

	cross_val_test_size=0.2
	bool_cnn = False
	fs_type = 'RFE'

	if type_classif == 'qda':
		kwargs = {"fs_type" : fs_type}

	elif type_classif == 'rf':
		kwargs = {"n_trees" : 100, "boots" : True, "fs_type" : fs_type}

	elif type_classif == 'svc':
		#kwargs = {"c" : 1.0, "kernel_type" : 'linear', "gamma_value" : 'scale'}
		kwargs = {"c" : 10.0, "kernel_type" : 'linear', "gamma_value" : 'scale',
		 "degree_value" : 3, "fs_type" : fs_type}
	elif type_classif == 'cnn':
		bool_cnn = True
		kwargs = {"Net_type" : 'mnist_net_dropout', "gpu" : False, "fs_type" : fs_type}
	else:
		raise ValueError("Select an aproppriate type of classifier")

	if split_type == 'all':

		num_splits = 5
		if type_crossval == 'k-fold':
			train_results = cv.k_fold_iter(train_data, train_labels, s_labels, train_ids, 
				num_ss=num_splits, clf=type_classif, is_cnn=bool_cnn, **kwargs)

		elif type_crossval == 'shuffleSplit':
			test_size = cross_val_test_size
			train_results = cv.shuffle_split_iter(train_data, train_labels, s_labels, 
				train_ids, num_ss=5, t_size=test_size, clf=type_classif, is_cnn=bool_cnn, 
				**kwargs)

		elif type_crossval == "final_test":
			test_data = test_dict['data']
			test_labels = test_dict['labels']
			test_ids = test_dict['ids']
			clf_function = cv.get_train_function(clf=type_classif)
			kwargs['fs_type'] = None
			train_results = [clf_function(train_data, train_labels, test_data, test_labels, **kwargs)]

		else:
			raise ValueError("If split_type: 'all'," +
				" you have to choose type_crossval: k-fold or shuffleSplit")

	if split_type == "general":

		train_grp_animals = train_dict['train_grps']
		print(train_grp_animals)
		n_groups = round(((1-test_percent) * 10)/2) 
		if type_crossval == 'groups_k-fold':
			train_results = cv.groups_k_fold_iter(train_data, train_labels, s_labels, train_ids,  
				train_grp_animals, num_groups=n_groups, clf=type_classif, 
				is_cnn=bool_cnn, **kwargs)

		elif type_crossval == 'L_P_Groups_out':
			n_groups = 2
			train_results = cv.leave_P_out_iter(train_data, train_labels, s_labels, train_ids, 
				train_grp_animals, num_groups=int(n_groups/2), clf=type_classif, is_cnn=bool_cnn, 
				**kwargs)

		elif type_crossval == "final_test":
			test_data = test_dict['data']
			test_labels = test_dict['labels']
			test_ids = test_dict['ids']
			clf_function = cv.get_train_function(clf=type_classif)
			kwargs['fs_type'] = None
			train_results = [clf_function(train_data, train_labels, test_data, test_labels, **kwargs)]

		else:
			raise ValueError("If split_type: 'general'," + 
			 " you have to choose type_crossval: groups_k-fold or L_P_Groups_out")

	
	print("\n ------------ Summary of results for fold 0 -----------------\n")
	print("# folds: ", len(train_results))
	print("test_labels shape: ", train_results[0]['clf_results'][0].shape) # test_labels
	print("predictions shape: ", train_results[0]['clf_results'][1].shape) # predictions
	print("preds: \n", train_results[0]['clf_results'][1]) 
	print("probabs or dec_function shape: ", train_results[0]['clf_results'][2].shape)
	print("probs or dec_function\n", train_results[0]['clf_results'][2]) # probabilities or dec_function
	print("Mean Acc: ", train_results[0]['clf_results'][3]) # mean acc

	if 'wrong_results' in list(train_results[0].keys()):
		print("wrong data shape: ", train_results[0]['wrong_results'][0].shape) # wrong data
		print("wrong string labs shape: " , train_results[0]['wrong_results'][1].shape) # wrong string labels "animal type"
		print("wrong ids shape: ", train_results[0]['wrong_results'][2].shape) # wrong ids
		print("wrong data: \n", train_results[0]['wrong_results'][0]) # wrong data
		print("wrong s labs: \n", train_results[0]['wrong_results'][1]) # wrong string labels "animal type"
		print("wrong ids: \n", train_results[0]['wrong_results'][2]) # wrong ids

	acc_values = np.zeros(len(train_results))
	for i in range(len(train_results)):
		acc_values[i] = train_results[i]['clf_results'][3]

	print("Mean acc for all folds:", np.mean(acc_values))

	#fs_results_manage(train_results, type_classif, kmers)

	#distr_missclass_data(kmers, train_results, type_crossval, type_classif, **kwargs)
	#sys.exit()

	get_metrics(train_results, type_crossval, type_classif, **kwargs)


if __name__ == '__main__':
	train_model(type_crossval="k-fold", type_classif="svc")
	#train_model(sys.argv[1], sys.argv[2])
