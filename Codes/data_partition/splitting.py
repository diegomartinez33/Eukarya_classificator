import numpy as np
import os
import sys
import sys
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold

# Code for file managemente library
biol_dir = "/hpcfs/home/da.martinez33/Biologia"
file_manage_folder = os.path.join(biol_dir,'Codes','file_management')
data_partition_folder = os.path.join(biol_dir,'Codes','data_partition')
sys.path.append(file_manage_folder)
sys.path.append(data_partition_folder)

from file_management import rws_files as rws
from data_partition import labeling as lb

real_names = ["stickleback", "whale_shark","bacalao","tilapia","salmon",
             "Acyrthosiphon","bombyx","Harpegnathos","locusta","tribolium"]

def split_data(counts_data,split_type="all",test_percent=0.2,type_labeling="per_group"):
    """ Function to split data in train and test datasets 
    counts_data: frequency data of kmers as a dictionary
    split_type:
    - "all": split each animal-data for train and test and merge all
    - "general": split data without mixing animals
    test_percent: percentage of data used to final test 
    type_labeling: defines the number of classes
    - "per_group": each broad group is a class (ex. fishes, insects, plants...)
        if just two groups are used, it would be a binary classification problem
    - "per_spp": each specimen is a class """
    train_part = {}
    test_part = {}
    train_data, test_data = [], []
    train_labels, test_labels = [], []

    print("Creating data partition... \n")
    if split_type == "all":
        num_animals = 0
        for animal in real_names:
            data = counts_data[animal]
            num_inst = int(data.shape[0])
            labels = [animal] * num_inst
            X_train, X_test, l_train, l_test = train_test_split(
                data, labels, test_size=test_percent, shuffle=True)
            if num_animals == 0:
                train_data, train_labels = X_train, l_train
                test_data, test_labels = X_test, l_test
            else:
                train_data = np.append(train_data,X_train,axis=0)
                train_labels.extend(l_train)
                test_data = np.append(test_data,X_test,axis=0)
                test_labels.extend(l_test)
            num_animals += 1

    elif split_type == "general":
        num_animals = 0
        for animal in real_names:
            if num_animals == 0:
                data = counts_data[animal]
                num_inst = int(data.shape[0 ])
                labels = [animal] * num_inst
            else:
                data = np.append(data,counts_data[animal],axis=0)
                num_inst = int(counts_data[animal].shape[0])
                labels.extend([animal] * num_inst)
            num_animals += 1
        labels = np.array(labels)

        groups_list, train_grp_animals = lb.grouping_partition(labels)
        gkf = GroupKFold(n_splits=2)
        for train_indx, test_indx in gkf.split(data, labels, groups=groups_list):
            train_data, test_data = data[train_indx], data[test_indx]
            train_labels, test_labels = labels[train_indx], labels[test_indx]
            s_labels = labels[train_indx]
    else:
        raise ValueError("You have to specify a valid split type in split_type parameter")

    print("Create numeric labels... ")
    train_labels = lb.create_num_labels(train_labels,type_labeling)
    test_labels = lb.create_num_labels(test_labels,type_labeling)

    train_part['data'] = train_data
    train_part['labels'] = train_labels
    test_part['data'] = test_data
    test_part['labels'] = test_labels
    if "s_labels" in locals():
        print("s_labels does exist")
        train_part['s_labs'] = s_labels
        train_part['train_grps'] = train_grp_animals
    return (train_part,test_part)




