import numpy as np
import os
import sys
import sys
import time
import random

## Violin plots
biol_dir = "/hpcfs/home/da.martinez33/Biologia"
real_names = ["stickleback", "whale_shark","bacalao","tilapia","salmon",
             "Acyrthosiphon","bombyx","Harpegnathos","locusta","tribolium"]
fishes = real_names[0:5]
insects = real_names[5:]
animal_groups = [fishes,insects]

def group_label(animalname,cont):
	num_label = -1
	for i, ani in enumerate(animal_groups):
		if animalname in ani:
			num_label = i
			break
	if num_label == -1:
		print(i)
		print(cont)
		print("animal name: ", animalname)
		print(ani)
		raise ValueError("Not assigned numeric label")
	return num_label

def grouping(label_data,test_percent=0.2):
	train_tip = 1
	test_tip = 2
	train_list = []
	test_list = []
	group_list = np.zeros(len(label_data))
	for group in animal_groups:
		num_samples = round(len(group) * (1 - 0.2))
		shuffled_list = group
		random.shuffle(shuffled_list)
		train_list.extend(shuffled_list[:num_samples])
		test_list.extend(shuffled_list[num_samples:])

	for cont, lab in enumerate(label_data):
		if lab in train_list:
			group_list[cont] = train_tip
		else:
			group_list[cont] = test_tip

	return group_list

def create_num_labels(label_data,labeling="per_group"):
    """ function that generates labels depending on type of classification 
    label_data: list with original labels (spp)
    labeling: defines the number of classes
    - "per_group": each broad group is a class (ex. fishes, insects, plants...)
        if just two groups are used, it would be a binary classification problem
    - "per_spp": each specimen is a class """
    numeric_labs = np.zeros(len(label_data))
    for cont, lab in enumerate(label_data):
    	if labeling == "per_group":
    		numeric_labs[cont] = group_label(lab,cont)
    	elif labeling == "per_spp":
    		numeric_labs[cont] = real_names.index(lab)
    	else:
    		raise ValueError("You have to specify a valid split type in labeling parameter")

    return numeric_labs




