import numpy as np
import os
import sys
import sys
import time

# Code for file managemente library
biol_dir = "/hpcfs/home/da.martinez33/Biologia"
file_manage_folder = os.path.join(biol_dir,'Codes','file_management')
cv_library_folder = os.path.join(biol_dir,'Codes','cv_lib')
data_partition_folder = os.path.join(biol_dir,'Codes','data_partition')
sys.path.append(file_manage_folder)
sys.path.append(cv_library_folder)
sys.path.append(data_partition_folder)

from file_management import get_names as gn
from file_management import rws_files as rws
from cv_lib import cv_ids
from data_partition import splitting

ani_names = ["stickleback", "whaleshark","Cod","tilapia","salmon",
             "acyroltosyphon","bombix","harpegrathos","locusta","tribolium"] 
real_names = ["stickleback", "whale_shark","bacalao","tilapia","salmon",
             "Acyrthosiphon","bombyx","Harpegnathos","locusta","tribolium"]
filter_cvs = True
filter_value = 0.95

data_dir = os.path.join(biol_dir,"Codes","counts")
results_dir = os.path.join(biol_dir,"Data","classification")
partitions_dir = os.path.join(results_dir,"data_partitions")
if os.path.isdir(results_dir) != True:
    os.mkdir(results_dir)

def data_reading():
    #Create dictionaries per animal
    print("Reading kmer counts...\n")
    animals = {} #Dictionary that contains counts for each animal
    kmers = [] #List of kmers
    cont = 0
    for file in os.listdir(data_dir):
        if file.endswith(".txt"):    
            file_name = file[:-4]
            animal = gn.get_animal(file)
            print(file_name)
            file_dir = os.path.join(data_dir,file)
            labels = np.genfromtxt(file_dir, skip_header=1, usecols=0, dtype=str)
            raw_data = np.genfromtxt(file_dir, skip_header=1)[:,1:]
            real_name = gn.get_real_animal_name(file_name)
            if filter_cvs:
                tsv_file = "cv_allReads_" + real_name + ".tsv"
                cvs_file = os.path.join(data_dir,"Scripts","cv_results",tsv_file)
                #raw_data: Datos a usar
                raw_data = cv_ids.remove_outliers(raw_data,
                    cv_ids.get_cv_idx(cvs_file,filter_value))

            #Diccionario: Es util guardarlo?
            #data = {label: row for label, row in zip(labels, raw_data)}
            # mini dict to save counts per kmer (label)
            if cont == 0:
                kmers = labels
            raw_data = np.transpose(raw_data)
            #Data divided 
            animals[real_name] = raw_data
            cont += 1
        if cont == len(ani_names):
            animals['kmers'] = kmers

    raw_data_file = os.path.join(results_dir,"counts_per_animal.pkl")
    rws.saveData(animals,raw_data_file)
    print("saved data in: \n",raw_data_file)

def data_splitting():
	split_type="all" #"all" or "general"
	test_percent=0.2
	type_labeling="per_group" #"per_group" or "per_spp"
	raw_data_file = os.path.join(results_dir,"counts_per_animal.pkl")
	counts_data = rws.loadData(raw_data_file)
	splitted = splitting.split_data(counts_data,split_type,test_percent,
		type_labeling)
	splitted_data_file = os.path.join(partitions_dir,
		"split_{}_{}_{}.pkl".format(split_type,str(test_percent),type_labeling))
	rws.saveData(splitted,splitted_data_file)

if __name__ == '__main__':
	if sys.argv[1] == "read":
		data_reading()
	elif sys.argv[1] == "data_splitting":
		data_splitting()

