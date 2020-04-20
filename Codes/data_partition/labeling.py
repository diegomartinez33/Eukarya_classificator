import numpy as np
import os
import sys
import sys
import time
import random

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

def grouping_partition(label_data, ani_gps=animal_groups, test_percent=0.2):
	train_tip = 1
	test_tip = 2
	train_list = []
	test_list = []
	group_list = np.zeros(len(label_data))
	train_group_list = []
	for group in ani_gps:
		num_samples = round(len(group) * (1 - test_percent))
		shuffled_list = group
		random.shuffle(shuffled_list)
		train_list.extend(shuffled_list[:num_samples])
		train_group_list.append(shuffled_list[:num_samples])
		test_list.extend(shuffled_list[num_samples:])

	for cont, lab in enumerate(label_data):
		if lab in train_list:
			group_list[cont] = train_tip
		else:
			group_list[cont] = test_tip

	return group_list, train_group_list

def grouping_crossval(label_data, ani_gps=animal_groups):
	train_tips = list(range(animal_groups[0]))
	group_list = np.zeros(len(label_data))
	shuffled_ani_list = []
	for group in animal_groups:
		shuffled_group = group
		random.shuffle(shuffled_group)
		shuffled_ani_list.append(np.array(shuffled_group))

	shuffled_ani_list = np.array(shuffled_ani_list)

	for cont, lab in enumerate(label_data):
		for i in shuffled_ani_list.shape[1]:
			if lab in shuffled_ani_list[:,i]:
				group_list[cont] = train_tips[i]
				break
			else:
				continue

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




